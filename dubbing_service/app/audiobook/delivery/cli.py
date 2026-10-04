"""Explicit operator operations: python -m ...delivery deliver --recipient brian --ready."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from .config import configured_path, enabled_recipient, load_config
from . import store


def prepare(config: dict, recipient_id: str) -> dict:
    from .projection import discover_reviewable
    enabled_recipient(config, recipient_id)
    manifests = sorted({Path(path).resolve() for pattern in config["export_manifests"] for path in glob.glob(pattern)})
    if not manifests:
        raise ValueError("No configured verified export manifests found")
    # Revision directories are retained. Old receipts are observations of old
    # state, not candidates; only receipts bound to today's canonical bytes enter
    # the stricter projection/QA gate below.
    current = {name: store.digest(Path(config["project_dir"]) / name)
               for name in ("project.json", "segments.json")}
    eligible = []
    for path in manifests:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("project_id") == Path(config["project_dir"]).name and receipt.get("rendition_state_hashes") == current:
            eligible.append(path)
    if not eligible:
        raise ValueError("No verified export is bound to the current Studio revision")
    projection = discover_reviewable(Path(config["project_dir"]), eligible)
    if config.get("book_title"):
        projection["book_title"] = str(config["book_title"])[:200]
    roots = [Path(root).resolve() for root in config["allowed_audio_roots"]]
    for chapter in projection["chapters"]:
        audio = Path(chapter["audio_path"]).resolve(strict=True)
        if not any(audio.is_relative_to(root) for root in roots):
            raise ValueError("Review audio is outside configured export roots")
    return store.publish_batch(Path(config["state_dir"]), projection, recipient_id)


def local_headers(config: dict) -> dict:
    return {"Tailscale-User-Login": enabled_recipient(config, "brian")["tailscale_login"]}


def verify_endpoint(config: dict, batch: dict, *, private: bool = True) -> dict:
    base = config["listening_base_url"] if private else f"http://127.0.0.1:{config['port']}"
    headers = {} if private else local_headers(config)
    # The external probe carries no forged identity: Tailscale must authenticate it.
    with httpx.Client(timeout=30, follow_redirects=False, trust_env=False, headers=headers) as client:
        health = client.get(base + "/health")
        health.raise_for_status()
        if health.json() != {"service": "audiobook-review", "status": "ok"}:
            raise ValueError("Unexpected listening service")
        response = client.get(base + "/api/batches/" + batch["batch_id"])
        response.raise_for_status()
        remote = response.json()
        if [(row["chapter_id"], row["audio_sha256"]) for row in remote["chapters"]] != [(row["chapter_id"],row["audio_sha256"]) for row in batch["chapters"]]:
            raise ValueError("Listening manifest differs from selected batch")
        for chapter in batch["chapters"]:
            response = client.get(base + f"/audio/{batch['batch_id']}/{chapter['chapter_id']}", headers={"Range": "bytes=0-1023"})
            if response.status_code != 206 or len(response.content) != 1024 or not response.headers.get("content-range", "").startswith("bytes 0-1023/"):
                raise ValueError("Audio Range verification failed")
            with Path(chapter["audio_path"]).open("rb") as source:
                if response.content != source.read(1024):
                    raise ValueError("Audio endpoint returned different bytes")
    return {"private_https" if private else "loopback": "PASS", "chapters": len(batch["chapters"]), "range": "PASS"}


def start_service(config: dict, config_path: Path) -> dict:
    url = f"http://127.0.0.1:{config['port']}/health"
    def healthy():
        try:
            response = httpx.get(url, headers=local_headers(config), timeout=2, trust_env=False)
            return response.status_code == 200 and response.json().get("service") == "audiobook-review"
        except (httpx.HTTPError, ValueError):
            return False
    if healthy():
        return {"service": "already_running"}
    state_dir = Path(config["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    with (state_dir / "service.log").open("ab") as output:
        process = subprocess.Popen([sys.executable, "-m", "dubbing_service.app.audiobook.delivery", "--config", str(config_path.resolve()), "serve"],
                                   cwd=Path(__file__).resolve().parents[4], stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=output, creationflags=flags,
                                   start_new_session=os.name != "nt")
    for _ in range(30):
        if healthy():
            return {"service": "started", "pid": process.pid}
        if process.poll() is not None:
            raise ValueError("Listening service could not start; inspect private service.log")
        time.sleep(.2)
    raise ValueError("Listening service startup timed out")


def taildrop(config: dict, batch: dict, *, force: bool = False) -> dict:
    recipient = enabled_recipient(config, batch["recipient_id"])
    target_dns = recipient.get("taildrop_dns")
    if not target_dns:
        return {"attempted": False, "state": "not_configured"}
    # Identity is checked immediately before transmission; never pick by an IP hint.
    status = subprocess.run(["tailscale", "status", "--json"], capture_output=True, check=True, text=True)
    status = json.loads(status.stdout)
    owner_id = status.get("Self", {}).get("UserID")
    matches = [peer for peer in status.get("Peer", {}).values()
               if peer.get("DNSName", "").rstrip(".") == target_dns.rstrip(".")
               and owner_id and peer.get("UserID") == owner_id
               and peer.get("OS") == "iOS" and peer.get("Online")
               and peer.get("TaildropTarget") == 1 and not peer.get("NoFileSharingReason")]
    if len(matches) != 1:
        return {"attempted": False, "state": "device_not_available", "destination": target_dns.split(".")[0]}
    state_dir = Path(config["state_dir"])
    ledger = state_dir / "taildrop"
    key = store.batch_identity(batch, recipient["id"])
    claim = ledger / (key + ".json")
    intent = {"batch_id": batch["batch_id"], "created_at": store.now(), "state": "pending", "destination": target_dns.split(".")[0]}
    if force:
        claim = ledger / (key + "-" + str(time.time_ns()) + ".json")
    if not store.write_once(claim, intent):
        return {"attempted": False, "state": "duplicate_suppressed", "destination": intent["destination"]}
    transfers = []
    transfer_attempted = False
    try:
        with ExitStack() as stack:
            snapshots = []
            # Taildrop consumes the verified handle, not an export path which
            # could have been replaced while waiting for the provider response.
            for chapter in batch["chapters"]:
                snapshot = stack.enter_context(tempfile.TemporaryFile())
                digest = hashlib.sha256()
                with Path(chapter["audio_path"]).open("rb") as source:
                    for block in iter(lambda: source.read(1024*1024), b""):
                        digest.update(block)
                        snapshot.write(block)
                if digest.hexdigest() != chapter["audio_sha256"]:
                    raise ValueError("Direct transfer audio changed")
                snapshot.seek(0)
                snapshots.append((chapter, snapshot))
            # Validate every file before the first transfer, and retain those
            # exact handles throughout the operation.
            for chapter, snapshot in snapshots:
                filename = f"Chapter_{chapter['chapter_number']}.mp3"
                transfer_attempted = True
                outcome = subprocess.run(["tailscale", "file", "cp", "--update-interval=0", "--name", filename, "-", target_dns.rstrip(".") + ":"],
                                         stdin=snapshot, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
                transfers.append({"chapter_id": chapter["chapter_id"], "audio_sha256": chapter["audio_sha256"],
                                  "exit_code": outcome.returncode, "state": "accepted_by_device" if outcome.returncode == 0 else "transfer_unknown"})
                if outcome.returncode != 0:
                    break
        result = {"attempted": True, "state": "accepted_by_device" if len(transfers) == len(batch["chapters"]) and all(row["exit_code"] == 0 for row in transfers) else "transfer_unknown", "destination": intent["destination"], "files": transfers}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        result = {"attempted": transfer_attempted, "state": "transfer_unknown", "destination": intent["destination"], "files": transfers}
    store.write_once(claim.with_name(claim.stem + "-outcome.json"), {**intent, **result, "completed_at": store.now()})
    store.record_event(state_dir, "direct_file_transfer", **result, batch_id=batch["batch_id"])
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Private audiobook review delivery")
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "deliver"):
        command = commands.add_parser(name)
        command.add_argument("--recipient", default="brian")
        if name == "deliver":
            command.add_argument("--ready", action="store_true", required=True)
            command.add_argument("--force", "--resend", action="store_true")
            command.add_argument("--taildrop", action="store_true")
    for name in ("serve", "start", "status", "feedback", "validate-notifications"):
        commands.add_parser(name)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        config_path = Path(args.config) if args.config else configured_path()
        if args.command == "serve":
            import uvicorn
            from .server import create_app
            uvicorn.run(create_app(config), host="127.0.0.1", port=config["port"], proxy_headers=False, access_log=False)
            return 0
        if args.command == "start":
            result = start_service(config, config_path)
        elif args.command == "feedback":
            result = {"feedback": store.list_feedback(Path(config["state_dir"]))}
        elif args.command == "validate-notifications":
            from .notify import validate_credentials
            result = validate_credentials(Path(config["notification_transport_root"]))
        elif args.command in {"prepare", "deliver"}:
            batch = prepare(config, args.recipient)
            url = config["listening_base_url"] + "/review/" + batch["batch_id"]
            result = {"batch_id": batch["batch_id"], "project_id": batch["project_id"], "chapters": len(batch["chapters"]), "listening_url": url}
            if args.command == "deliver":
                result["service"] = start_service(config, config_path)
                result["local_verification"] = verify_endpoint(config, batch, private=False)
                result["private_verification"] = verify_endpoint(config, batch, private=True)
                if prepare(config, args.recipient)["batch_id"] != batch["batch_id"]:
                    raise ValueError("Canonical selection changed before notification")
                from .notify import send_notification
                result["notification"] = send_notification(batch, enabled_recipient(config, args.recipient), state_dir=Path(config["state_dir"]),
                                                           transport_root=Path(config["notification_transport_root"]), listening_url=url, force=args.force)
                if args.taildrop:
                    result["direct_files"] = taildrop(config, batch, force=args.force)
                store.record_event(Path(config["state_dir"]), "delivery_operation", **result)
        else:
            result = {"project_id": Path(config["project_dir"]).name, "listening_url": config["listening_base_url"],
                      "batches": len(list((Path(config["state_dir"])/"manifests").glob("*.json"))),
                      "feedback_count": len(store.list_feedback(Path(config["state_dir"]))), "enabled_recipients": [name for name,recipient in config["recipients"].items() if recipient.get("enabled")]}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        # Never stringify arbitrary provider/configuration exceptions; they can carry credentials.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "detail": "Operation stopped safely. Check private configuration and verification evidence."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
