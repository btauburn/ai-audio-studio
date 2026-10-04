"""Private operator configuration; credentials remain in the existing transport store."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

CONFIG_ENV = "AUDIOBOOK_DELIVERY_CONFIG"
DEFAULT_CONFIG = Path.home() / ".config" / "audiobook-delivery" / "config.json"


def configured_path() -> Path:
    """The operator configuration this deployment uses.

    A machine can host more than one Studio against more than one projects root,
    and each needs its own delivery identity; ``AUDIOBOOK_DELIVERY_CONFIG``
    selects it without moving the default or copying credentials.
    """
    selected = os.environ.get(CONFIG_ENV)
    return Path(selected).expanduser() if selected and selected.strip() else DEFAULT_CONFIG


def load_config(path: Path | None = None) -> dict:
    config = json.loads(Path(path or configured_path()).read_text(encoding="utf-8-sig"))
    project_dir = Path(config["project_dir"]).resolve(strict=True)
    if not (project_dir / "project.json").is_file():
        raise ValueError("Configured Studio project is missing")
    config["project_dir"] = str(project_dir)
    config["state_dir"] = str(project_dir / "delivery")
    config.setdefault("port", 8778)
    if not isinstance(config["port"], int) or not 1024 <= config["port"] <= 65535:
        raise ValueError("Invalid listening port")
    url = urlsplit(config["listening_base_url"])
    if url.scheme != "https" or not url.hostname or not url.hostname.endswith(".ts.net") or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        raise ValueError("Listening URL must use private Tailscale HTTPS")
    config["listening_base_url"] = config["listening_base_url"].rstrip("/")
    studio = config.get("studio_base_url")
    if studio is not None:
        url = urlsplit(studio)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("The Studio this projection reads must be a loopback HTTP origin")
        config["studio_base_url"] = studio.rstrip("/")
    recipients = config.get("recipients", {})
    if set(recipients) != {"brian", "mk"} or recipients["mk"].get("enabled") is not False:
        raise ValueError("V1 requires Brian and a disabled MK configuration")
    for name, recipient in recipients.items():
        recipient["id"] = name
        if recipient.get("enabled") and (name != "brian" or not recipient.get("tailscale_login")):
            raise ValueError("Only Brian's configured Tailscale identity is authorized")
    config["allowed_audio_roots"] = [str(Path(root).resolve(strict=True)) for root in config["allowed_audio_roots"]]
    if not config["allowed_audio_roots"]:
        raise ValueError("Explicit audio roots required")
    return config


def enabled_recipient(config: dict, recipient_id: str) -> dict:
    recipient = config["recipients"].get(recipient_id)
    if not recipient or not recipient.get("enabled") or recipient_id != "brian":
        raise ValueError("Recipient is disabled or unauthorized")
    return recipient
