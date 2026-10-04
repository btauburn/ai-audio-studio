"""Voice Lab routes, sharing Studio's canonical project and job owner."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.audiobook.delivery.server import verified_audio_response
from dubbing_service.app.storage.projects import ProjectStorageError

router = APIRouter(prefix="/projects/{project_id}/voice-lab")


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Preference(Input):
    key: str = Field(max_length=100)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    favorite: bool | None = None
    rating: int | None = Field(default=None, ge=1, le=5, strict=True)
    suitability: int | None = Field(default=None, ge=1, le=5, strict=True)
    role_id: str | None = Field(default=None, max_length=100)


class Position(Input):
    key: str | None = Field(default=None, max_length=100)
    audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    seconds: float = Field(default=0, ge=0)
    filter: str | None = Field(default=None, pattern="^(new|favorites|rated|all)$")


class Match(Input):
    description: str = Field(default="", max_length=2000)
    role_id: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=5, ge=1, le=5)


class MoreLike(Input):
    key: str = Field(max_length=100)
    request_id: str = Field(min_length=1, max_length=100)


class Assignment(Input):
    key: str = Field(max_length=100)
    role_id: str = Field(max_length=100)
    expected_identity_id: str | None = Field(max_length=100)


class Automation(Input):
    paused: bool


class Design(Input):
    role_id: str = Field(max_length=100)
    candidate_id: str = Field(max_length=100)
    expected_identity_id: str | None = Field(max_length=100)
    request_id: str = Field(min_length=1, max_length=100)


class ManifestInput(Input):
    manifest: dict
    source_text: str = Field(max_length=500000)
    expected_manifest_id: str | None = Field(default=None, max_length=100)


class ManifestVersion(Input):
    expected_manifest_id: str = Field(max_length=100)


class ReferenceInput(Input):
    role_id: str = Field(max_length=100)
    acquisition_id: str = Field(max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    acquisition_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_identity_id: str | None = Field(max_length=100)
    request_id: str = Field(min_length=1, max_length=100)


@router.get("/references")
def reference_list(project_id: str):
    from dubbing_service.app.audiobook.reference_transcripts import list_acquisitions
    return call(list_acquisitions, project_id)


@router.api_route("/references/{acquisition_id}/audio", methods=["GET", "HEAD"])
def reference_audio(project_id: str, acquisition_id: str, acquisition_sha256: str, request: Request):
    from dubbing_service.app.audiobook.reference_transcripts import acquisition_audio
    path, checksum = call(acquisition_audio, project_id, acquisition_id, acquisition_sha256)
    return verified_audio_response(path, checksum, request, 1, False)


class ReferenceTranscript(Input):
    acquisition_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReferenceQuality(ReferenceTranscript):
    expected_text: str | None = Field(default=None, min_length=1, max_length=1000)


@router.post("/references/{acquisition_id}/prepare")
def reference_prepare(project_id: str, acquisition_id: str, payload: ReferenceQuality):
    from dubbing_service.app.audiobook.reference_preparation import prepare_acquisition
    return call(prepare_acquisition, project_id, acquisition_id, **payload.model_dump())


@router.post("/references/{acquisition_id}/quality")
def reference_quality(project_id: str, acquisition_id: str, payload: ReferenceQuality):
    from dubbing_service.app.audiobook.reference_quality import acquisition_quality
    return call(acquisition_quality, project_id, acquisition_id, **payload.model_dump())


class ReferenceTranscriptCorrection(ReferenceTranscript):
    text: str = Field(min_length=1, max_length=1000)
    expected_text_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    expected_correction_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


@router.post("/references/{acquisition_id}/transcript")
def reference_transcript(project_id: str, acquisition_id: str, payload: ReferenceTranscript):
    from dubbing_service.app.audiobook.reference_transcripts import acquisition_transcript
    return call(acquisition_transcript, project_id, acquisition_id, payload.acquisition_sha256)


@router.post("/references/{acquisition_id}/transcript/correction")
def reference_transcript_correction(project_id: str, acquisition_id: str, payload: ReferenceTranscriptCorrection):
    from dubbing_service.app.audiobook.reference_transcripts import correct_acquisition_transcript
    return call(correct_acquisition_transcript, project_id, acquisition_id, **payload.model_dump())


class LibraryPublish(Input):
    key: str = Field(max_length=100)


class LibraryImport(Input):
    library_voice_id: str = Field(max_length=100)
    version: str = Field(max_length=100)


class SpeakerCorrection(BaseModel):
    # Additional audit fields are inert evidence retained with the derivation.
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    utterance_id: str = Field(min_length=1, max_length=100)
    role_id: str = Field(min_length=1, max_length=100)
    source_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reason: str = Field(min_length=1, max_length=4000)
    evidence_utterance_ids: list[str] = Field(default_factory=list, max_length=50)
    from_role_id: str | None = Field(default=None, max_length=100)


class AttributionReview(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    source_project_id: str = Field(min_length=1, max_length=100)
    chapter_id: str = Field(min_length=1, max_length=100)
    source_revision_id: str = Field(min_length=1, max_length=100)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corrections: list[SpeakerCorrection] = Field(default_factory=list, max_length=2000)
    narration_fragment: SpeakerCorrection | None = None
    alias_corrections: dict[str, str] = Field(default_factory=dict)


class ChapterFork(Input):
    chapter_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)
    attribution_review: AttributionReview | None = None


class ChapterRender(Input):
    chapter_id: str = Field(min_length=1, max_length=100)
    segment_ids: list[str] | None = Field(default=None, min_length=1, max_length=2000)
    request_id: str = Field(min_length=1, max_length=100)


class ChapterExport(Input):
    chapter_id: str = Field(min_length=1, max_length=100)


class Issue(Input):
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seconds: float = Field(ge=0)
    text: str = Field(min_length=1, max_length=4000)
    category: str = Field(pattern="^(technical|performance)$")


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except ProjectStorageError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("")
def get_lab(project_id: str):
    return call(lab.view, project_id)


class CastingPrepare(Input):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    utterance_ids: list[str] | None = Field(default=None, min_length=1, max_length=20000)


class CastingRound(Input):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    role_id: str = Field(min_length=1, max_length=100)
    expected_round_revision: int = Field(ge=0, strict=True)
    expected_assignment_revision: int = Field(ge=0, strict=True)
    intent: Literal["generate", "none_fit"]


class CastingFeedback(Input):
    event_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    role_id: str = Field(min_length=1, max_length=100)
    candidate_id: str = Field(min_length=1, max_length=100)
    round_revision: int = Field(ge=1, strict=True)
    assignment_revision: int = Field(ge=0, strict=True)
    key: str = Field(max_length=100)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stars: int | None = Field(default=None, ge=1, le=5, strict=True)
    selection_confirmed: bool | None = Field(default=None, strict=True)
    raw_direction: str = Field(default="", max_length=4000)
    action: Literal["save", "use", "recast", "preview"] = "save"
    baseline_candidate_ids: list[str] = Field(default_factory=list, max_length=10)


class CastingRender(CastingPrepare):
    utterance_ids: list[str] = Field(min_length=1, max_length=20000)


class CastingResume(Input):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")


class CastingResolve(Input):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    mode: Literal["suggested", "narrator", "skip", "auto"]


class CastingSkip(Input):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    utterance_ids: list[str] = Field(min_length=1, max_length=100)


class CastingRestore(Input):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    reason: Literal["unknown_speaker", "unspeakable", "failed_qa", "human"] | None = None
    utterance_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)


class CastingPolicy(Input):
    enabled: bool


class CastingAttribute(Input):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    assignments: dict[str, str] = Field(min_length=1, max_length=1200)
    new_roles: list[str] = Field(default_factory=list, max_length=25)
    aliases: dict[str, str] = Field(default_factory=dict, max_length=25)
    note: str | None = Field(default=None, max_length=200)


@router.get("/casting/book")
def casting_book(project_id: str):
    """Where the book is: voices waiting, recording progress, finished chapters."""
    from dubbing_service.app.audiobook import autocast_book
    return call(autocast_book.book_status, project_id)


@router.post("/casting/production/resolve")
def casting_resolve(project_id: str, payload: CastingResolve, tasks: BackgroundTasks):
    """The human's explicit decision for lines that have no speaker."""
    from dubbing_service.app.audiobook import autocast_book, casting_review
    result = call(autocast_book.resolve_unattributed, project_id, **payload.model_dump())
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return {"decision": result, "book": call(autocast_book.book_status, project_id)}


@router.post("/casting/production/skip")
def casting_skip(project_id: str, payload: CastingSkip, tasks: BackgroundTasks):
    """Leave chosen lines out of the recording."""
    from dubbing_service.app.audiobook import autocast_book, casting_review
    result = call(autocast_book.skip_lines, project_id, **payload.model_dump())
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return {"decision": result, "book": call(autocast_book.book_status, project_id)}


@router.post("/casting/production/restore")
def casting_restore(project_id: str, payload: CastingRestore, tasks: BackgroundTasks):
    """Bring skipped lines back into the recording."""
    from dubbing_service.app.audiobook import autocast_book, casting_review
    result = call(autocast_book.restore_lines, project_id, **payload.model_dump())
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return {"decision": result, "book": call(autocast_book.book_status, project_id)}


@router.post("/casting/production/policy")
def casting_policy(project_id: str, payload: CastingPolicy, tasks: BackgroundTasks):
    """Turn automatic handling of lines that would stop the book on or off."""
    from dubbing_service.app.audiobook import autocast_book, casting_review
    result = call(autocast_book.set_automatic_handling, project_id, payload.enabled)
    if payload.enabled:
        tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return {"decision": result, "book": call(autocast_book.book_status, project_id)}


@router.post("/casting/production/attribute")
def casting_attribute(project_id: str, payload: CastingAttribute, tasks: BackgroundTasks):
    """Attribute the speakers of lines. Studio only: the phone-facing service does not offer it."""
    from dubbing_service.app.audiobook import autocast_book, casting_review
    result = call(autocast_book.attribute_lines, project_id, **payload.model_dump())
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return {"decision": result, "book": call(autocast_book.book_status, project_id)}


@router.api_route("/casting/chapters/{export_id}/audio", methods=["GET", "HEAD"])
def casting_chapter_audio(project_id: str, export_id: str, request: Request, download: bool = False):
    from dubbing_service.app.audiobook import autocast_book
    path, sha256, _label, number = call(autocast_book.chapter_artifact, project_id, export_id)
    return verified_audio_response(path, sha256, request, number, download)


@router.get("/casting")
def casting_view(project_id: str):
    from dubbing_service.app.audiobook import casting_review
    return call(casting_review.view, project_id)


@router.post("/casting/prepare")
def casting_prepare(project_id: str, payload: CastingPrepare, tasks: BackgroundTasks):
    from dubbing_service.app.audiobook import casting_review
    result = call(casting_review.prepare, project_id, **payload.model_dump())
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return result


@router.post("/casting/round")
def casting_round(project_id: str, payload: CastingRound, tasks: BackgroundTasks):
    from dubbing_service.app.audiobook import casting_review
    result = call(casting_review.request_round, project_id, **payload.model_dump())
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return result


@router.post("/casting/feedback")
def casting_feedback(project_id: str, payload: CastingFeedback, tasks: BackgroundTasks):
    from dubbing_service.app.audiobook import casting_review
    from fastapi.responses import JSONResponse
    result = call(casting_review.feedback, project_id, payload.model_dump(exclude_none=True))
    if result.get("conflict"):
        return JSONResponse(result, status_code=409)
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return result


@router.post("/casting/render")
def casting_render(project_id: str, payload: CastingRender, tasks: BackgroundTasks):
    from dubbing_service.app.audiobook import casting_review
    result = call(casting_review.render_request, project_id, **payload.model_dump())
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return result


@router.post("/casting/resume")
def casting_resume(project_id: str, payload: CastingResume, tasks: BackgroundTasks):
    from dubbing_service.app.audiobook import casting_review
    result = call(casting_review.resume_request, project_id, payload.request_id)
    tasks.add_task(casting_review.maintenance_tick, project_ids=[project_id])
    return result


@router.post("/fork-chapter")
def fork_chapter(project_id: str, payload: ChapterFork):
    from dubbing_service.app.audiobook.voice_lab_chapter import fork_chapter as fork
    return call(fork, project_id, **payload.model_dump())


@router.post("/render-chapter")
def render_chapter(project_id: str, payload: ChapterRender, tasks: BackgroundTasks):
    from dubbing_service.app.audiobook.voice_lab_chapter import render_chapter as render
    return call(render, project_id, **payload.model_dump(), schedule=tasks.add_task)


@router.post("/export-chapter")
def export_chapter(project_id: str, payload: ChapterExport):
    from dubbing_service.app.audiobook.voice_lab_chapter_export import export_chapter as export
    return call(export, project_id, **payload.model_dump())


@router.post("/preferences")
def preference(project_id: str, payload: Preference):
    return call(lab.save_preference, project_id, payload.model_dump(exclude_unset=True))


@router.post("/position")
def position(project_id: str, payload: Position):
    return call(lab.save_position, project_id, payload.model_dump(exclude_none=True))


@router.post("/match")
def match(project_id: str, payload: Match):
    return call(lab.match, project_id, **payload.model_dump())


@router.post("/more-like")
def more_like(project_id: str, payload: MoreLike):
    return call(lab.more_like, project_id, **payload.model_dump())


@router.post("/assign")
def assign(project_id: str, payload: Assignment):
    return call(lab.assign, project_id, **payload.model_dump())


@router.post("/automation")
def automation(project_id: str, payload: Automation, tasks: BackgroundTasks):
    result = call(lab.set_automation, project_id, payload.paused)
    if not payload.paused:
        tasks.add_task(lab.discovery_tick, project_id)
    return result


@router.post("/issues")
def issue(project_id: str, payload: Issue):
    return call(lab.add_issue, project_id, payload.model_dump())


@router.post("/design")
def design(project_id: str, payload: Design):
    return call(lab.design, project_id, **payload.model_dump())


@router.post("/source-preflight")
def source_preflight(project_id: str):
    return call(lab.source_preflight, project_id)


@router.post("/excerpt-manifest")
def excerpt_manifest(project_id: str, payload: ManifestInput):
    from dubbing_service.app.audiobook.voice_lab_render import persist_excerpt_manifest
    return call(persist_excerpt_manifest, project_id, **payload.model_dump())


@router.post("/prepare-cast")
def prepare_cast(project_id: str, payload: ManifestVersion):
    return call(lab.prepare_listening_cast, project_id, **payload.model_dump())


@router.post("/render-excerpt")
def render_excerpt(project_id: str, payload: ManifestVersion):
    from dubbing_service.app.audiobook.voice_lab_render import render_listening_excerpt
    return call(render_listening_excerpt, project_id, **payload.model_dump())


@router.post("/import-reference")
def import_reference(project_id: str, payload: ReferenceInput):
    return call(lab.import_reference, project_id, **payload.model_dump())


@router.get("/library")
def library(project_id: str):
    from dubbing_service.app.audiobook.voice_lab_library import list_versions
    call(lab.view, project_id)
    return {"voices": call(list_versions)}


@router.post("/library/publish")
def library_publish(project_id: str, payload: LibraryPublish):
    from dubbing_service.app.audiobook.voice_lab_library import publish_version
    return call(publish_version, project_id, payload.key)


@router.post("/library/import")
def library_import(project_id: str, payload: LibraryImport):
    from dubbing_service.app.audiobook.voice_lab_library import import_version
    return call(import_version, project_id, **payload.model_dump())


@router.api_route("/audio/{key}", methods=["GET", "HEAD"])
def audio(project_id: str, key: str, request: Request):
    path, record = call(lab.audio_record, project_id, key)
    return verified_audio_response(path, record["audio_sha256"], request, 1, False)


@router.get("/runtime-comparison")
def runtime_comparison(project_id: str):
    from dubbing_service.app.audiobook.runtime_comparison import load_comparison
    return call(load_comparison, project_id)


@router.api_route("/runtime-comparison/audio/{pair_id}/{lane}", methods=["GET", "HEAD"])
def runtime_comparison_audio(project_id: str, pair_id: str, lane: str, request: Request):
    from dubbing_service.app.audiobook.runtime_comparison import comparison_audio
    path, checksum = call(comparison_audio, project_id, pair_id, lane)
    return verified_audio_response(path, checksum, request, 1, False)



class PreferredMigration(Input):
    dry_run: bool = True
    expected_plan_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


@router.post("/library/migrate-preferences")
def migrate_preferences(project_id: str, payload: PreferredMigration):
    from dubbing_service.app.audiobook.preferred_voice_migration import migrate
    return call(migrate, project_id, **payload.model_dump())


class PreferredMigrationRollback(Input):
    migration_id: str = Field(pattern=r"^migration_[a-f0-9]{32}$")


@router.post("/library/migrate-preferences/rollback")
def rollback_preferences(project_id: str, payload: PreferredMigrationRollback):
    from dubbing_service.app.audiobook.preferred_voice_migration import rollback
    return call(rollback, project_id, payload.migration_id)



class QualitativeFeedback(Input):
    event_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    role_id: str = Field(min_length=1, max_length=100)
    candidate_id: str = Field(min_length=1, max_length=100)
    key: str = Field(min_length=1, max_length=100)
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_feedback: str = Field(min_length=1, max_length=4000)
    requested_performer: str = Field(min_length=1, max_length=200)


@router.post("/casting/quality-feedback")
def qualitative_feedback(project_id: str, payload: QualitativeFeedback):
    from dubbing_service.app.audiobook.casting_review import qualitative_rejection
    return call(qualitative_rejection, project_id, **payload.model_dump())
