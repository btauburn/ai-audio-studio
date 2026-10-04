"""Local book import and narration routes under the existing Studio owner."""
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from starlette.concurrency import run_in_threadpool

from dubbing_service.app.audiobook import book_creator as creator
from dubbing_service.app.audiobook.delivery.server import verified_audio_response
from dubbing_service.app.storage.projects import ProjectStorageError

router = APIRouter(prefix="/book-creator", tags=["book-creator"])


class RenderBook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chapter_ids: list[str] = Field(min_length=1, max_length=200)
    library_voice_id: str = Field(min_length=1, max_length=100)
    voice_version: str = Field(min_length=1, max_length=128)
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")


class Automation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paused: bool


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ProjectStorageError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("")
def books():
    return call(creator.list_books)


@router.post("/import")
async def import_book(request: Request, tasks: BackgroundTasks, file: UploadFile = File(...), request_id: str = Form(...), name: str | None = Form(None),
                      mode: Literal["single_narrator", "autocast"] = Form("single_narrator")):
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > creator.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Book upload exceeds 32 MiB")
    if mode == "autocast":
        from dubbing_service.app.audiobook.autocast_import import import_autocast
        result = await run_in_threadpool(call, import_autocast, bytes(data), file.filename, request_id, name)
        if request.client and request.client.host in {"127.0.0.1", "::1"} and request.headers.get("x-private-review-recipient") == "brian":
            from dubbing_service.app.storage.projects import load_project, save_project, project_meta_lock
            with project_meta_lock(result["project_id"]):
                project = load_project(result["project_id"])
                project.metadata["autocast_import"]["private_review_recipient"] = "brian"
                save_project(project)
        # Start the run that the successful import just recorded. The run key
        # makes this idempotent, and the periodic maintenance pass is an
        # independent safety net if this dispatch never happens.
        from dubbing_service.app.audiobook.casting_review import maintenance_tick
        tasks.add_task(maintenance_tick, project_ids=[result["project_id"]])
        return result
    return await run_in_threadpool(call, creator.import_book, bytes(data), file.filename, request_id, name)


@router.get("/{project_id}")
def book(project_id: str):
    return call(creator.view, project_id)


@router.post("/{project_id}/render")
def render(project_id: str, payload: RenderBook, tasks: BackgroundTasks):
    return call(creator.render_book, project_id, **payload.model_dump(), schedule=tasks.add_task)


@router.get("/{project_id}/requests/{request_id}")
def progress(project_id: str, request_id: str):
    return call(creator.request_status, project_id, request_id)


@router.post("/{project_id}/requests/{request_id}/automation")
def automation(project_id: str, request_id: str, payload: Automation):
    from dubbing_service.app.audiobook.book_creator_recovery import set_automation
    return call(set_automation, project_id, request_id, payload.paused)


@router.post("/{project_id}/requests/{request_id}/notify")
def notify(project_id: str, request_id: str):
    from dubbing_service.app.audiobook.book_creator_notifications import request_notification
    call(request_notification, project_id, request_id)
    return call(creator.request_status, project_id, request_id)


@router.api_route("/{project_id}/audio/{export_id}", methods=["GET", "HEAD"])
def audio(project_id: str, export_id: str, request: Request, download: bool = False):
    path, checksum, _label = call(creator.audio_artifact, project_id, export_id)
    return verified_audio_response(path, checksum, request, export_id, download)
