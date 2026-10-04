"""CPU-only verification-cache contracts, including real Windows sharing."""
from __future__ import annotations

import asyncio
import hashlib
import json
import mmap
import os
import sys
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from service import pinned_model_verification as pinned
from scripts.model_preflight import ManifestError

windows = pytest.mark.skipif(os.name != "nt", reason="Win32 file-share guarantees")


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    def make(name="model", *, invalid_hash=False):
        folder = root / name
        folder.mkdir()
        artifacts = []
        for index in range(2):
            file = folder / f"weight{index}.bin"
            data = bytes([index + 1]) * 256
            file.write_bytes(data)
            artifacts.append({"path": file.relative_to(root).as_posix(), "size": len(data),
                              "sha256": "0" * 64 if invalid_hash else hashlib.sha256(data).hexdigest(),
                              "source": "fixture"})
        manifest = root / f"{name}.json"
        manifest.write_text(json.dumps({"schema_version": 1, "name": name, "source": "fixture",
            "revision": "fixture-1", "sources": {"fixture": {"provider": "fixture"}}, "files": artifacts}), encoding="utf-8")
        return manifest, [root / artifact["path"] for artifact in artifacts]
    return root, make


@pytest.fixture
def cache():
    instance = pinned.PinnedModelVerificationCache()
    yield instance
    instance.clear()


def count_hashes(monkeypatch):
    hashes = []
    original = pinned.verify_artifact
    def counted(*args, **kwargs):
        if kwargs.get("verify_hash", True):
            hashes.append(args[1]["path"])
        return original(*args, **kwargs)
    monkeypatch.setattr(pinned, "verify_artifact", counted)
    return hashes


def writable(paths):
    for path in paths:
        original = path.read_bytes()
        path.write_bytes(original)


@windows
def test_windows_cold_hashes_all_then_reuses_locked_success(model, cache, monkeypatch):
    root, make = model
    manifest, artifacts = make()
    hashes = count_hashes(monkeypatch)
    first = cache.verify(manifest, project_root=root)
    assert first.pinned and not first.cache_hit
    second = cache.verify(manifest, project_root=root)
    assert second.pinned and second.cache_hit
    assert len(hashes) == 2
    assert all(status.ok for status in second.artifacts)
    first.manifest["files"][0]["sha256"] = "0" * 64
    assert cache.verify(manifest, project_root=root).manifest["files"][0]["sha256"] != "0" * 64


@windows
def test_read_pins_deny_write_delete_replace_and_parent_rename_then_clear(model, cache):
    root, make = model
    manifest, artifacts = make()
    cache.verify(manifest, project_root=root)
    for path in [manifest, *artifacts]:
        with pytest.raises(OSError):
            path.write_bytes(path.read_bytes())
        with pytest.raises(OSError):
            path.rename(path.with_suffix(".moved"))
        with pytest.raises(OSError):
            path.unlink()
    replacement = root / "replacement.bin"
    replacement.write_bytes(artifacts[0].read_bytes())
    with pytest.raises(OSError):
        replacement.replace(artifacts[0])
    with pytest.raises(OSError):
        artifacts[0].parent.rename(root / "renamed-model")
    cache.clear()
    cache.clear()  # idempotent cleanup
    writable([manifest, *artifacts])
    replacement.replace(artifacts[0])
    artifacts[0].rename(artifacts[0].with_suffix(".moved"))


@windows
@pytest.mark.parametrize("failure", ["hash", "size", "schema", "missing"])
def test_failed_verification_releases_every_new_handle(model, cache, failure):
    root, make = model
    manifest, artifacts = make(invalid_hash=failure == "hash")
    if failure == "size":
        artifacts[0].write_bytes(b"short")
    elif failure == "schema":
        manifest.write_text('{"schema_version":2}', encoding="utf-8")
    elif failure == "missing":
        artifacts[1].unlink()
    with pytest.raises(ValueError):
        cache.verify(manifest, project_root=root)
    assert not cache._entries
    writable([manifest, *[file for file in artifacts if file.exists()]])


@windows
def test_pin_failure_falls_back_to_full_hash_every_time_and_releases_partial_pins(model, cache, monkeypatch):
    root, make = model
    manifest, artifacts = make()
    hashes = count_hashes(monkeypatch)
    original = pinned._ReadPin
    def unavailable(path):
        if path == artifacts[1]:
            raise OSError("fixture sharing conflict after partial acquisition")
        return original(path)
    monkeypatch.setattr(pinned, "_ReadPin", unavailable)
    for _ in range(2):
        result = cache.verify(manifest, project_root=root)
        assert not result.pinned and not result.cache_hit
        assert not cache._entries
        writable([manifest, *artifacts])
    assert len(hashes) == 4


@windows
def test_existing_writer_prevents_cache_and_same_size_edit_is_reverified(model, cache):
    root, make = model
    manifest, artifacts = make()
    with artifacts[0].open("r+b") as writer:
        result = cache.verify(manifest, project_root=root)
        assert not result.pinned and not result.cache_hit
        writer.seek(0)
        writer.write(b"x" * 256)
        writer.flush()
        with pytest.raises(ValueError, match="integrity"):
            cache.verify(manifest, project_root=root)
    assert not cache._entries
    writable([manifest, *artifacts])


@windows
def test_existing_writable_mapping_prevents_read_pin_and_cached_trust(model, cache):
    root, make = model
    manifest, artifacts = make()
    with artifacts[0].open("r+b") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE)
    try:
        with pytest.raises(OSError):
            pinned._ReadPin(artifacts[0])
        result = cache.verify(manifest, project_root=root)
        assert not result.pinned and not result.cache_hit
        mapping[0:1] = b"x"
        mapping.flush()
        with pytest.raises(ValueError, match="integrity"):
            cache.verify(manifest, project_root=root)
    finally:
        mapping.close()
    assert not cache._entries
    writable([manifest, *artifacts])


def test_non_windows_fallback_always_fully_verifies(model, cache, monkeypatch):
    root, make = model
    manifest, artifacts = make()
    monkeypatch.setattr(pinned, "_windows_pins_available", lambda: False)
    hashes = count_hashes(monkeypatch)
    for _ in range(2):
        result = cache.verify(manifest, project_root=root)
        assert not result.cache_hit and not result.pinned
    assert len(hashes) == 4 and not cache._entries
    artifacts[0].write_bytes(b"x" * 256)
    with pytest.raises(ValueError, match="integrity"):
        cache.verify(manifest, project_root=root)


@windows
def test_uncacheable_component_falls_back_without_retained_handles(model, cache, monkeypatch):
    root, make = model
    manifest, artifacts = make()
    original = pinned._cacheable_path
    monkeypatch.setattr(pinned, "_cacheable_path", lambda path, base: False if path == artifacts[1] else original(path, base))
    result = cache.verify(manifest, project_root=root)
    assert not result.pinned and not result.cache_hit
    assert not cache._entries
    writable([manifest, *artifacts])


@windows
def test_path_escape_policy_is_preserved_and_manifest_pin_released(model, cache):
    root, make = model
    manifest, artifacts = make()
    payload = json.loads(manifest.read_text())
    payload["files"][0]["path"] = "../outside.bin"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="escapes"):
        cache.verify(manifest, project_root=root)
    assert not cache._entries
    writable([manifest, *artifacts])


@windows
def test_file_identity_mismatch_invalidates_success_and_rehashes(model, cache, monkeypatch):
    root, make = model
    manifest, _ = make()
    hashes = count_hashes(monkeypatch)
    cache.verify(manifest, project_root=root)
    entry = next(iter(cache._entries.values()))
    entry.pins[manifest].identity = (0, 0, 0)
    result = cache.verify(manifest, project_root=root)
    assert not result.cache_hit and result.pinned
    assert len(hashes) == 4


@windows
def test_clear_allows_explicit_updates_but_bad_new_bytes_are_rejected(model, cache):
    root, make = model
    manifest, artifacts = make()
    cache.verify(manifest, project_root=root)
    cache.clear()
    artifacts[0].write_bytes(b"x" * 256)
    with pytest.raises(ValueError, match="integrity"):
        cache.verify(manifest, project_root=root)
    writable([manifest, *artifacts])


@windows
def test_multiple_model_entries_survive_switches_and_all_release_on_clear(model, cache, monkeypatch):
    root, make = model
    first, first_artifacts = make("base")
    second, second_artifacts = make("voice_design")
    hashes = count_hashes(monkeypatch)
    cache.verify(first, project_root=root)
    cache.verify(second, project_root=root)
    assert cache.verify(first, project_root=root).cache_hit
    assert cache.verify(second, project_root=root).cache_hit
    assert len(hashes) == 4 and len(cache._entries) == 2
    cache.clear()
    writable([first, second, *first_artifacts, *second_artifacts])


@windows
def test_entry_bound_evicts_and_closes_oldest_handles(model, cache):
    root, make = model
    entries = [make(f"model{index}") for index in range(5)]
    for manifest, _ in entries:
        cache.verify(manifest, project_root=root)
    assert len(cache._entries) == 4
    writable([entries[0][0], *entries[0][1]])
    with pytest.raises(OSError):
        entries[-1][0].write_bytes(entries[-1][0].read_bytes())


@pytest.fixture
def server(monkeypatch):
    from service import qwen3tts_server
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float16="float16", bfloat16="bfloat16", float32="float32",
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(
        Qwen3TTSModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())))
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace())
    monkeypatch.setattr(qwen3tts_server, "qwen3tts_model", None)
    monkeypatch.setattr(qwen3tts_server, "_active_model_kind", None)
    monkeypatch.setattr(qwen3tts_server, "_active_attention", None)
    monkeypatch.setattr(qwen3tts_server, "_model_lock", Lock())
    monkeypatch.setattr(qwen3tts_server, "_infer_lock", Lock())
    return qwen3tts_server


@windows
def test_server_internal_switch_keeps_pins_explicit_unload_closes(model, cache, server, monkeypatch):
    root, make = model
    manifests = root / "models/manifests"
    manifests.mkdir(parents=True)
    base, base_artifacts = make("base")
    design, design_artifacts = make("design")
    base = base.rename(manifests / "qwen3_tts_base.json")
    design = design.rename(manifests / "qwen3_tts_voicedesign.json")
    monkeypatch.setattr(server, "project_root", str(root))
    monkeypatch.setattr(server, "_model_verification_cache", cache)
    monkeypatch.setattr(server, "config", SimpleNamespace(base_model_dir="base", model_dir="design", dtype="bfloat16", device="cpu"))
    monkeypatch.setattr(server, "_validated_model_path", lambda args: root / "design")
    monkeypatch.setattr(server, "require_project_model_dir", lambda *args, **kwargs: root / "base")
    monkeypatch.setattr(server.gc, "collect", lambda: None)
    hashes = count_hashes(monkeypatch)
    for kind in ["base", "voice_design", "base", "voice_design"]:
        server._load_qwen3tts_model_if_needed(kind, attention="sdpa")
    assert len(hashes) == 4 and len(cache._entries) == 2
    for manifest in [base, design]:
        with pytest.raises(OSError):
            manifest.write_bytes(manifest.read_bytes())
    assert server.unload_model()["success"]
    assert not cache._entries and server.qwen3tts_model is None
    writable([base, design, *base_artifacts, *design_artifacts])


@windows
def test_lifespan_exception_releases_verification_pins(model, cache, server, monkeypatch):
    root, make = model
    manifest, artifacts = make()
    cache.verify(manifest, project_root=root)
    monkeypatch.setattr(server, "_model_verification_cache", cache)
    monkeypatch.setattr(server, "_parse_server_args", lambda: SimpleNamespace(outputs_keep_days=0, outputs_max_files=0))
    monkeypatch.setattr(server, "_validated_model_path", lambda args: root)
    monkeypatch.chdir(root)
    async def run():
        with pytest.raises(RuntimeError, match="fixture shutdown error"):
            async with server.lifespan(server.app):
                raise RuntimeError("fixture shutdown error")
    asyncio.run(run())
    assert not cache._entries
    writable([manifest, *artifacts])


def test_close_failure_still_closes_other_pins_and_other_entries(cache):
    calls = []
    class Pin:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail
        def close(self):
            calls.append(self.name)
            if self.fail:
                raise OSError("fixture close error")
    first = pinned._Entry({}, (), {Path("first"): Pin("first")})
    second = pinned._Entry({}, (), {Path("second"): Pin("second"), Path("bad"): Pin("bad", fail=True)})
    cache._entries[(Path("root"), Path("first"))] = first
    cache._entries[(Path("root"), Path("second"))] = second
    with pytest.raises(OSError, match="fixture close error"):
        cache.clear()
    assert set(calls) == {"first", "second", "bad"}
    assert not cache._entries and not first.pins and not second.pins


@windows
def test_native_writable_view_blocks_pinning_after_all_file_and_mapping_handles_close(model, cache):
    import ctypes
    from ctypes import wintypes
    root, make = model
    manifest, artifacts = make()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    def bind(name, args, result):
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, result
        return function
    create = bind("CreateFileW", [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE], wintypes.HANDLE)
    create_mapping = bind("CreateFileMappingW", [wintypes.HANDLE, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.LPCWSTR], wintypes.HANDLE)
    map_view = bind("MapViewOfFile", [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, ctypes.c_size_t], wintypes.LPVOID)
    unmap = bind("UnmapViewOfFile", [wintypes.LPCVOID], wintypes.BOOL)
    close = bind("CloseHandle", [wintypes.HANDLE], wintypes.BOOL)
    flush = bind("FlushViewOfFile", [wintypes.LPCVOID, ctypes.c_size_t], wintypes.BOOL)
    file_handle = create(str(artifacts[0]), 0xC0000000, 3, None, 3, 0x80, None)
    assert file_handle != ctypes.c_void_p(-1).value
    mapping = view = None
    try:
        mapping = create_mapping(file_handle, None, 4, 0, 0, None)  # PAGE_READWRITE
        assert mapping
        view = map_view(mapping, 2, 0, 0, 256)  # FILE_MAP_WRITE
        assert view
        assert close(file_handle)
        file_handle = None
        with pytest.raises(OSError):
            pinned._ReadPin(artifacts[0])
        assert close(mapping)
        mapping = None  # Native view alone remains; no duplicated Python file handle.
        with pytest.raises(OSError):
            pinned._ReadPin(artifacts[0])
        result = cache.verify(manifest, project_root=root)
        assert not result.pinned and not result.cache_hit
        ctypes.memmove(view, b"x", 1)
        assert flush(view, 256)
        assert artifacts[0].read_bytes()[0:1] == b"x"
        with pytest.raises(ValueError, match="integrity"):
            cache.verify(manifest, project_root=root)
    finally:
        if view:
            assert unmap(view)
        if mapping:
            assert close(mapping)
        if file_handle:
            assert close(file_handle)
    assert not cache._entries
    writable([manifest, *artifacts])


@windows
def test_in_root_junction_lexical_path_always_uses_full_verification(model, cache, monkeypatch):
    import subprocess
    root, make = model
    manifest, artifacts = make()
    junction = root / "model-junction"
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(artifacts[0].parent)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    try:
        assert getattr(junction.lstat(), "st_file_attributes", 0) & 0x400
        payload = json.loads(manifest.read_text())
        for item in payload["files"]:
            item["path"] = "model-junction/" + Path(item["path"]).name
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        # Existing policy safely resolves this in-root junction; it remains a
        # valid full-verification path but cannot authorize pinned hash reuse.
        assert pinned.artifact_target(payload, payload["files"][0], root) == artifacts[0]
        hashes = count_hashes(monkeypatch)
        for _ in range(2):
            verified = cache.verify(manifest, project_root=root)
            assert not verified.pinned and not verified.cache_hit
        assert len(hashes) == 4 and not cache._entries
        writable([manifest, *artifacts])
    finally:
        junction.resolve().relative_to(root)  # verified disposable target scope
        junction.rmdir()  # Remove the junction itself; retain target fixtures.
    assert all(path.is_file() for path in artifacts)
