"""Recovery uses isolated canonical fixtures only; no GPU/network/push."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_book_creator import source, store, imported, render  # noqa: F401
from tests.audiobook_helpers import make_tone_wav
from dubbing_service.app.models.project import SegmentTake
from dubbing_service.app.audiobook import book_creator as creator, book_creator_recovery as recovery
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.models.jobs import SynthesisJobStatus
from dubbing_service.app.routers.book_creator import router

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def fail_render(store, monkeypatch, *, dispatched=False, error="bounded QA fixture"):
    def failure(project_id, ids, **kwargs):
        return [creator.pipeline.RenderRecord(segment_id=ids[0], take_id="", success=False, selected=False,
                    qa_passed=False, dispatched=dispatched, error=error)]
    monkeypatch.setattr(creator.pipeline, "render_utterances", failure)
    book = imported(store)
    status = render(store, book)
    assert status["status"] == "interrupted"
    return book, status


def run_scheduled(tasks):
    fn, args = tasks.pop(0)
    # Owner-supplied test renderer prevents even readiness calls; fixture
    # pipeline and encoder use isolated signal data, not TTS or actual MP3.
    return fn(*args[:-1], renderer=SimpleNamespace(backend="qwen3tts"))


def test_new_receipt_defaults_and_get_has_no_writes_or_dispatch(store):
    book = imported(store); tasks=[]
    status = render(store, book, schedule=lambda fn,*args: tasks.append((fn,args)))
    assert status["automation"]["enabled"] and status["notify_on_completion"]
    root = store[0][0]/book["project_id"]
    before = {p.name:p.read_bytes() for p in root.glob("*.json")}
    creator.request_status(book["project_id"], "render-one"); creator.view(book["project_id"])
    assert before == {p.name:p.read_bytes() for p in root.glob("*.json")}
    assert not store[2]["generated"] and len(tasks)==1
    run_scheduled(tasks)


def test_transient_recovery_backoff_one_claim_resume_without_duplicate_valid_takes(store, monkeypatch):
    normal=creator.pipeline.render_utterances
    book,status=fail_render(store, monkeypatch)
    tasks=[];schedule=lambda fn,*args:tasks.append((fn,args))
    first=recovery.maintenance_tick(now=NOW,schedule=schedule)
    assert first["dispatched"]==0 and not tasks
    assert creator.request_status(book["project_id"],"render-one")["automation"]["next_retry_at"]==(NOW+timedelta(seconds=30)).isoformat()
    recovery.maintenance_tick(now=NOW+timedelta(seconds=29),schedule=schedule)
    assert not tasks
    result=recovery.maintenance_tick(now=NOW+timedelta(seconds=30),schedule=schedule)
    assert result["dispatched"]==1 and len(tasks)==1
    recovery.maintenance_tick(now=NOW+timedelta(days=5),schedule=schedule)
    assert len(tasks)==1
    monkeypatch.setattr(creator.pipeline,"render_utterances",normal)
    final=run_scheduled(tasks)
    assert final["status"]=="succeeded" and final["automation"]["retry_count"]==1
    assert len(store[2]["generated"])==len(set(store[2]["generated"]))==4


def test_exactly_two_retries_30_then_120_and_no_default_scheduler(store,monkeypatch):
    book,status=fail_render(store,monkeypatch);tasks=[];schedule=lambda fn,*args:tasks.append((fn,args))
    recovery.maintenance_tick(now=NOW)
    due=recovery.maintenance_tick(now=NOW+timedelta(seconds=30))
    assert due["due"]==1 and not creator.is_job_active(book["project_id"],status["job_id"])
    recovery.maintenance_tick(now=NOW+timedelta(seconds=30),schedule=schedule);run_scheduled(tasks)
    recovery.maintenance_tick(now=NOW+timedelta(seconds=31),schedule=schedule)
    retry=creator.request_status(book["project_id"],"render-one")["automation"]
    assert retry["retry_count"]==1 and retry["next_retry_at"]==(NOW+timedelta(seconds=151)).isoformat()
    recovery.maintenance_tick(now=NOW+timedelta(seconds=150),schedule=schedule);assert not tasks
    recovery.maintenance_tick(now=NOW+timedelta(seconds=151),schedule=schedule);run_scheduled(tasks)
    recovery.maintenance_tick(now=NOW+timedelta(days=1),schedule=schedule)
    final=creator.request_status(book["project_id"],"render-one")["automation"]
    assert final["retry_count"]==2 and final["state"]=="exhausted" and not tasks


def test_historical_receipts_not_auto_enabled_or_notified(store,monkeypatch):
    book,_=fail_render(store,monkeypatch)
    p=load_project(book["project_id"]);r=p.metadata["book_creator"]["requests"]["render-one"]
    r.pop("automation");r.pop("notify_on_completion");save_project(p)
    path=store[0][0]/p.id/"project.json";before=path.read_bytes()
    result=recovery.maintenance_tick(now=NOW,schedule=lambda *args:pytest.fail("historical dispatch"))
    assert not result["dispatched"] and path.read_bytes()==before
    status=creator.request_status(p.id,"render-one")
    assert not status["automation"]["enabled"] and not status["notify_on_completion"]


def test_pause_endpoint_survives_reload_and_resumes_future_dispatch(store,monkeypatch):
    book,_=fail_render(store,monkeypatch)
    app=FastAPI();app.include_router(router,prefix="/api/v1")
    with TestClient(app) as client:
        url=f"/api/v1/book-creator/{book['project_id']}/requests/render-one/automation"
        assert client.post(url,json={"paused":True}).json()["automation"]["paused"]
        recovery.maintenance_tick(now=NOW,schedule=lambda *args:pytest.fail("paused dispatch"))
        assert creator.request_status(book["project_id"],"render-one")["automation"]["paused"]
        assert not client.post(url,json={"paused":False}).json()["automation"]["paused"]
        assert client.post(url,json={"paused":False,"unexpected":1}).status_code==422


def test_pause_during_line_finishes_it_then_stops_before_next(store,monkeypatch):
    book=imported(store);normal=creator.pipeline.render_utterances
    def pause_after_one(pid,ids,**kwargs):
        result=normal(pid,ids,**kwargs);recovery.set_automation(pid,"render-one",True);return result
    monkeypatch.setattr(creator.pipeline,"render_utterances",pause_after_one)
    status=render(store,book)
    assert status["status"]=="interrupted" and status["completed"]==1
    assert len(store[2]["generated"])==1 and not status["outputs"]


@pytest.mark.parametrize("change",["cancelled","stop","source","cast","selection"])
def test_cancel_stale_and_changed_selection_never_auto_resume(store,monkeypatch,change):
    book,status=fail_render(store,monkeypatch);pid=book["project_id"]
    if change=="cancelled":creator.mutate_job(pid,status["job_id"],lambda j:setattr(j,"status",SynthesisJobStatus.CANCELLED))
    elif change=="stop":creator.mutate_job(pid,status["job_id"],lambda j:setattr(j,"stop_requested",True))
    elif change=="cast":
        p=load_project(pid);p.roles[0].voice_identity.id="newer-cast";save_project(p)
    else:
        segments=load_segments(pid)
        if change=="source":segments[0].text+=" changed"
        else:
            path=make_tone_wav(store[0][0]/pid/"audio/owner-selected.wav",seconds=.5)
            take=SegmentTake(id="newer-owner-selection",audio_file="audio/owner-selected.wav",audio_sha256=creator.file_sha256(path))
            segments[0].takes.append(take);segments[0].selected_take_id=take.id
        save_segments(pid,segments)
    recovery.maintenance_tick(now=NOW,schedule=lambda *args:pytest.fail("unsafe dispatch"))
    state=creator.request_status(pid,"render-one")["automation"]
    assert state["state"]=="blocked" and state["retry_count"]==0


def test_production_priority_does_not_consume_retry(store,monkeypatch):
    book,status=fail_render(store,monkeypatch)
    lease=creator.try_register_running_batch(store[0][2].id)
    try:
        recovery.maintenance_tick(now=NOW,schedule=lambda *args:pytest.fail("production overlap"))
        state=creator.request_status(book["project_id"],"render-one")["automation"]
        assert state["state"]=="waiting_for_production" and state["retry_count"]==0
    finally:creator.unregister_running_batch(lease)


@pytest.mark.parametrize("error",["timed out","connection lost","unclassified failure",""])
def test_unknown_dispatched_transport_outcome_blocks_duplicates(store,monkeypatch,error):
    book,status=fail_render(store,monkeypatch,dispatched=True,error=error)
    recovery.maintenance_tick(now=NOW,schedule=lambda *args:pytest.fail("uncertain duplicate"))
    state=creator.request_status(book["project_id"],"render-one")["automation"]
    assert state["state"]=="blocked" and "uncertain" in state["reason"]


def test_crash_after_valid_take_commit_reconciles_and_reuses_exact_audio(store,monkeypatch):
    normal=creator.pipeline.render_utterances;book=imported(store)
    def crash_after_commit(pid,ids,**kwargs):
        normal(pid,ids,**kwargs);raise KeyboardInterrupt("after canonical Take commit")
    monkeypatch.setattr(creator.pipeline,"render_utterances",crash_after_commit)
    with pytest.raises(KeyboardInterrupt):render(store,book)
    pid=book["project_id"];segment=load_segments(pid)[0];take_id=segment.selected_take_id
    assert take_id and len(store[2]["generated"])==1
    recovery.maintenance_tick(now=NOW)
    assert load_project(pid).metadata["book_creator"]["requests"]["render-one"]["inflight_utterance_id"] is None
    tasks=[];recovery.maintenance_tick(now=NOW+timedelta(seconds=30),schedule=lambda fn,*args:tasks.append((fn,args)))
    monkeypatch.setattr(creator.pipeline,"render_utterances",normal)
    assert run_scheduled(tasks)["status"]=="succeeded"
    assert load_segments(pid)[0].selected_take_id==take_id
    assert len(store[2]["generated"])==len(set(store[2]["generated"]))==4


def test_live_active_request_not_taken_over_after_any_elapsed_time(store):
    book=imported(store);tasks=[]
    state=render(store,book,schedule=lambda fn,*args:tasks.append((fn,args)))
    recovery.maintenance_tick(now=NOW+timedelta(days=365),schedule=lambda *args:pytest.fail("live takeover"))
    assert len(tasks)==1 and creator.request_status(book["project_id"],"render-one")["automation"]["retry_count"]==0
    run_scheduled(tasks)


def test_cancel_during_export_is_not_overwritten_by_success(store,monkeypatch):
    book=imported(store,data=b"A single original line.");normal=creator._export
    def cancel_after_export(pid,request_id,chapter_ids,binding):
        result=normal(pid,request_id,chapter_ids,binding)
        status=creator.request_status(pid,request_id)
        creator.mutate_job(pid,status["job_id"],lambda j:setattr(j,"status",SynthesisJobStatus.CANCELLED))
        return result
    monkeypatch.setattr(creator,"_export",cancel_after_export)
    final=render(store,book)
    assert final["status"]=="cancelled" and len(final["outputs"])==1
    recovery.maintenance_tick(now=NOW,schedule=lambda *args:pytest.fail("cancelled retry"))
    assert creator.request_status(book["project_id"],"render-one")["automation"]["state"]=="blocked"


def test_committed_but_corrupt_inflight_take_cannot_clear_uncertainty(store,monkeypatch):
    normal=creator.pipeline.render_utterances;book=imported(store)
    def crash_after_commit(pid,ids,**kwargs):
        normal(pid,ids,**kwargs);raise KeyboardInterrupt("crash fixture")
    monkeypatch.setattr(creator.pipeline,"render_utterances",crash_after_commit)
    with pytest.raises(KeyboardInterrupt):render(store,book)
    segment=load_segments(book["project_id"])[0]
    take=creator.takes.selected_take(segment)
    (store[0][0]/book["project_id"]/take.audio_file).write_bytes(b"corrupt isolated fixture")
    recovery.maintenance_tick(now=NOW,schedule=lambda *args:pytest.fail("corrupt retry"))
    receipt=load_project(book["project_id"]).metadata["book_creator"]["requests"]["render-one"]
    assert receipt["inflight_utterance_id"]==segment.id
    assert receipt["automation"]["state"]=="blocked"


def test_notification_projection_retains_safe_retry_and_acceptance_evidence():
    receipt={"notify_on_completion":True,"notification":{"state":"accepted","detail":"Provider accepted",
        "attempts":1,"next_retry_at":None,"accepted":True,"device_delivery_confirmed":False,
        "review_url":"/voice-lab.html?home=1","export_ids":["mp3_fixture"],"secret":"hidden"}}
    result=recovery.public_notification(receipt)
    assert result["status"]=="accepted" and result["message"]=="Provider accepted"
    assert result["accepted"] and not result["device_delivery_confirmed"] and "secret" not in result


def test_manual_resume_cannot_clear_unknown_remote_request(store,monkeypatch):
    book,_=fail_render(store,monkeypatch,dispatched=True,error="connection lost")
    monkeypatch.setattr(creator.pipeline,"render_utterances",lambda *args,**kwargs:pytest.fail("duplicate provider request"))
    with pytest.raises(creator.BookCreatorError,match="outcome is uncertain"):
        render(store,book)
    receipt=load_project(book["project_id"]).metadata["book_creator"]["requests"]["render-one"]
    assert receipt["inflight_utterance_id"]


@pytest.mark.parametrize("project_count,requests_per_project,target_project,target_request,passes",[
    (65,1,64,0,2),(2,100,1,99,2),(2,70,1,64,3)])
def test_bounded_fair_scan_reaches_later_projects_and_requests(tmp_path,monkeypatch,project_count,requests_per_project,target_project,target_request,passes):
    projects={};dispatched=[]
    for index in range(project_count):
        pid=f"project-{index:03d}";folder=tmp_path/pid;folder.mkdir();(folder/"project.json").write_text("{}")
        requests={}
        for row in range(requests_per_project):
            opted=index==target_project and row==target_request
            requests[f"request-{row:03d}"]={"job_id":f"job-{row}","chapter_ids":["chapter"],"library_voice_id":"voice",
                "voice_version":"version","automation":{"enabled":opted,"paused":False,"retry_count":0,
                "next_retry_at":(NOW-timedelta(seconds=1)).isoformat()}}
        projects[pid]=SimpleNamespace(id=pid,metadata={"book_creator":{"import_status":"ready","requests":requests}})
    monkeypatch.setattr(recovery,"_project_after",None);monkeypatch.setattr(recovery,"_request_after",{})
    monkeypatch.setattr(recovery,"get_config",lambda:SimpleNamespace(projects_root_abs=tmp_path))
    monkeypatch.setattr(recovery,"load_project",lambda pid:projects[pid])
    monkeypatch.setattr(recovery,"get_job",lambda *args:SimpleNamespace(stop_requested=False,status=SimpleNamespace(value="interrupted"),message="fixture"))
    monkeypatch.setattr(recovery,"is_job_active",lambda *args:False)
    monkeypatch.setattr(recovery,"has_active_tts_operation",lambda:False)
    monkeypatch.setattr(creator,"_pending_production",lambda *args:False)
    monkeypatch.setattr(creator,"_source",lambda pid:(projects[pid],projects[pid].metadata["book_creator"],[]))
    monkeypatch.setattr(recovery,"retry_guard",lambda *args:None)
    monkeypatch.setattr(creator,"render_book",lambda pid,**kwargs:(dispatched.append((pid,kwargs["request_id"])) or {"automation":{"state":"retrying"}}))
    for index in range(passes):
        result=recovery.maintenance_tick(now=NOW,schedule=lambda *args:None)
        assert result["scanned"]<=100 and result["dispatched"]<=1
        if index<passes-1:assert not dispatched
    assert dispatched==[(f"project-{target_project:03d}",f"request-{target_request:03d}")]


def test_two_safe_inactive_running_books_resume_serially_after_restart(store,monkeypatch):
    normal=creator.pipeline.render_utterances
    first,first_status=fail_render(store,monkeypatch)
    second=imported(store,request_id="import-two");second_status=render(store,second)
    for book,status in [(first,first_status),(second,second_status)]:
        creator.mutate_job(book["project_id"],status["job_id"],lambda j:setattr(j,"status",SynthesisJobStatus.RUNNING))
        assert recovery.recoverable_idle_book_job(book["project_id"],creator.get_job(book["project_id"],status["job_id"]))
        assert not creator._pending_production(book["project_id"],status["job_id"])
    tasks=[];schedule=lambda fn,*args:tasks.append((fn,args))
    recovery.maintenance_tick(now=NOW,schedule=schedule)
    recovery.maintenance_tick(now=NOW+timedelta(seconds=30),schedule=schedule)
    assert len(tasks)==1
    active_pid=tasks[0][1][0]
    waiting=second if first["project_id"]==active_pid else first
    waiting_status=creator.request_status(waiting["project_id"],"render-one")
    assert creator._pending_production(waiting["project_id"],waiting_status["job_id"])
    recovery.maintenance_tick(now=NOW+timedelta(seconds=30),schedule=schedule)
    assert len(tasks)==1
    monkeypatch.setattr(creator.pipeline,"render_utterances",normal)
    assert run_scheduled(tasks)["status"]=="succeeded"
    recovery.maintenance_tick(now=NOW+timedelta(seconds=31),schedule=schedule)
    assert len(tasks)==1
    assert run_scheduled(tasks)["status"]=="succeeded"
    assert len(store[2]["generated"])==8


@pytest.mark.parametrize("peer_state",["safe","active","uncertain","ordinary","paused","exhausted","stale"])
def test_cold_worker_priority_uses_same_readonly_safe_peer_proof(store,monkeypatch,peer_state):
    from dubbing_service.app.audiobook import voice_lab_runtime as runtime
    from dubbing_service.app.routers import settings
    book,status=fail_render(store,monkeypatch);pid=book["project_id"];job_id=status["job_id"]
    creator.mutate_job(pid,job_id,lambda j:setattr(j,"status",SynthesisJobStatus.RUNNING))
    if peer_state=="active":creator.mark_job_active(pid,job_id,True)
    elif peer_state=="ordinary":creator.mutate_job(pid,job_id,lambda j:setattr(j,"job_type","single_synthesize"))
    elif peer_state in {"uncertain","paused","exhausted","stale"}:
        project=load_project(pid);receipt=project.metadata["book_creator"]["requests"]["render-one"]
        if peer_state=="uncertain":receipt["inflight_utterance_id"]=receipt["segment_ids"][0]
        elif peer_state=="paused":receipt["automation"]["paused"]=True
        elif peer_state=="exhausted":receipt["automation"]["retry_count"]=2
        else:project.roles[0].voice_identity.id="newer-owner-cast"
        save_project(project)
    root=store[0][0]/pid;before={p.name:p.read_bytes() for p in root.glob("*.json")}
    health=iter([False,False,True])
    monkeypatch.setattr(runtime,"_healthy",lambda _:next(health))
    monkeypatch.setattr(settings,"_get_tts_runtime_state",lambda:(False,None,None))
    starts=[];monkeypatch.setattr(settings,"api_start_tts_service",lambda payload:starts.append(payload))
    try:
        if peer_state=="safe":
            runtime.ensure_backend_ready("qwen3tts",job_id="new-owner-job")
            assert len(starts)==1 and starts[0].persist_default is False
        else:
            with pytest.raises(ValueError,match="priority"):
                runtime.ensure_backend_ready("qwen3tts",job_id="new-owner-job")
            assert not starts
        assert before=={p.name:p.read_bytes() for p in root.glob("*.json")}
    finally:
        if peer_state=="active":creator.mark_job_active(pid,job_id,False)


def test_malformed_peer_never_receives_priority_exemption():
    assert not recovery.recoverable_idle_book_job("missing",SimpleNamespace(id="ordinary",status="running"))
