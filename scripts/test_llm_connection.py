"""Minimal credential-safe live check for configured DeepSeek V4 presets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing_service.app.core.config import get_config  # noqa: E402
from dubbing_service.app.providers.llm import OpenAIStyleLLMProvider  # noqa: E402
from dubbing_service.app.storage.llm_providers import load_llm_providers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-pro", action="store_true", help="also call the configured Pro fallback")
    args = parser.parse_args()

    providers = load_llm_providers()
    models = json.loads((ROOT / "dubbing_service" / "llm_models.json").read_text(encoding="utf-8"))
    provider_by_id = {item.id: item for item in providers}
    targets = ["deepseek-v4-flash"]
    if args.include_pro:
        targets.append("deepseek-v4-pro")

    failures = 0
    for model_name in targets:
        config = next((item for item in models if item.get("model") == model_name), None)
        if not config:
            print(f"{model_name}: FAIL missing model preset")
            failures += 1
            continue
        configured_provider_id = config.get("provider_id")
        if configured_provider_id:
            provider_config = provider_by_id.get(configured_provider_id)
        else:
            provider_config = next(
                (item for item in providers if item.is_default),
                providers[0] if providers else None,
            )

        if provider_config is not None:
            base_url = provider_config.base_url
            api_key = provider_config.api_key
        else:
            global_config = get_config()
            base_url = global_config.llm_base_url
            api_key = global_config.llm_api_key

        if not base_url or not api_key:
            print(f"{model_name}: FAIL missing local credential")
            failures += 1
            continue

        provider = OpenAIStyleLLMProvider(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            max_attempts=1,
        )
        result = provider.chat(
            [
                {"role": "system", "content": "只输出 JSON 对象。"},
                {"role": "user", "content": '返回 {"ok":true}。'},
            ],
            max_tokens=128,
            response_format={"type": "json_object"},
            thinking={"type": "disabled"},
            max_attempts=1,
        )
        valid_json = False
        if result.text:
            try:
                valid_json = isinstance(json.loads(result.text), dict)
            except json.JSONDecodeError:
                valid_json = False
        ok = result.success and valid_json
        print(
            json.dumps(
                {
                    "model": model_name,
                    "success": ok,
                    "status_code": result.status_code,
                    "finish_reason": result.finish_reason,
                    "request_id_present": bool(result.request_id),
                    "valid_json": valid_json,
                    "usage": result.usage,
                },
                ensure_ascii=False,
            )
        )
        failures += 0 if ok else 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
