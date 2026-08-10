# License scope map

This repository is a multi-license distribution. It is not licensed as a
single Apache-2.0 work. Copyright in the platform-specific work identified
below is held by **TadashiJLU**. Upstream and third-party components retain
their own copyright and license terms.

The rules below are applied from the most specific row to the least specific
row. A license file or copyright notice stored beside a component always takes
precedence over this summary.

## Path-to-license boundary

| Path or provenance boundary | License for that material | Authoritative notice |
| --- | --- | --- |
| `dubbing_service/**`, except `dubbing_service/webui/vendor/**` | Apache-2.0 for TadashiJLU-authored platform code and documentation | `LICENSES/Apache-2.0.txt` |
| `service/**`, `scripts/**`, `environments/**` | Apache-2.0 for TadashiJLU-authored platform integration and operations code | `LICENSES/Apache-2.0.txt` |
| `models/README.md`, `models/manifests/**` | Apache-2.0 for the manifest schema, metadata curation and documentation only; this does **not** license any model weight | `LICENSES/Apache-2.0.txt`, each manifest's `sources.*.license` |
| `start_platform.bat`, `tools/platform_launcher.py`, `indextts/utils/audio_safety.py` | Apache-2.0 | `LICENSES/Apache-2.0.txt` |
| `tests/__init__.py`, `tests/audio_test_utils.py`, `tests/test_*.py` | Apache-2.0 | `LICENSES/Apache-2.0.txt` |
| `docs/**` | Apache-2.0 for TadashiJLU-authored project documentation | `LICENSES/Apache-2.0.txt` |
| `README.md`, `README_EN.md`, `SECURITY.md`, `NOTICE.md`, `THIRD_PARTY_NOTICES.md` | Apache-2.0 for TadashiJLU-authored text; quoted or incorporated upstream material keeps its original license | `LICENSES/Apache-2.0.txt` and the relevant upstream notice |
| `archive/README.md`, `archive/AI_AUDIO_STUDIO_PRODUCTION_PLAN.md`, `archive/LEGACY_QUICKSTART_ZH.md` | Apache-2.0 for TadashiJLU-authored project documentation | `LICENSES/Apache-2.0.txt` |
| `archive/README_SERVICE_LEGACY.md`, `archive/LEGACY_DUBBING_PLATFORM_GUIDE.md` | Apache-2.0 only for TadashiJLU-authored portions; no relicensing is asserted for embedded upstream excerpts | `LICENSES/Apache-2.0.txt` and the relevant upstream notice |
| Files that existed in upstream IndexTTS2 commit `1698b32033f38a034572891aed698609da2ff392`, including `indextts/**`, `webui.py`, `assets/**`, `examples/**`, `tools/**` except `tools/platform_launcher.py`, `tests/cases.jsonl`, `tests/padding_test.py`, `tests/regression_test.py`, and `tests/sample_prompt.wav` | bilibili IndexTTS2 Model Use License; later modifications to those files remain derivatives under that license unless an individual file clearly says otherwise | `LICENSES/Bilibili-IndexTTS.txt` and `LICENSES/Bilibili-IndexTTS-ZH.txt` |
| `archive/README_INDEXTTS_1_5.md`, `archive/README_ROOT_WITH_UPSTREAM_REFERENCE.md`, `archive/UPSTREAM_README_ZH.md` | Upstream IndexTTS material / derivative documentation under the bilibili IndexTTS2 Model Use License; no Apache relicensing is asserted | `LICENSES/Bilibili-IndexTTS.txt` and `LICENSES/Bilibili-IndexTTS-ZH.txt` |
| `cosyvoice/**`, except `cosyvoice/third_party/Matcha-TTS/**` and files carrying their own notice | Apache-2.0 under the CosyVoice upstream copyright and notices; local modifications remain subject to Apache-2.0 | `cosyvoice/LICENSE`, `docs/third_party_sources.md` |
| `cosyvoice/third_party/Matcha-TTS/**` | MIT, except any nested file with its own license | `cosyvoice/third_party/Matcha-TTS/LICENSE` |
| `cosyvoice/third_party/Matcha-TTS/matcha/hifigan/**` | MIT under its retained notice | `cosyvoice/third_party/Matcha-TTS/matcha/hifigan/LICENSE` |
| `dubbing_service/webui/vendor/tailwindcss-3.4.17/**` | MIT | adjacent `LICENSE` |
| `dubbing_service/webui/vendor/vue-3.5.39/**` | MIT | adjacent `LICENSE` |
| `dubbing_service/webui/vendor/wavesurfer-7.12.10/**` | BSD-3-Clause | adjacent `LICENSE` |

## Mixed and unclassified files

Repository-level files that combine upstream material with later platform
changes (for example build metadata and lockfiles) are not silently
relicensed. The pre-existing portion keeps its upstream license; only an
original TadashiJLU contribution is offered under Apache-2.0. If the boundary
cannot be established from Git history and this map, treat the file as
unclassified and obtain a provenance review before redistribution.

No bulk copyright headers have been added to mixed or upstream-derived files.
The upstream baseline is deliberately recorded so provenance can be checked
with Git, for example:

```text
git cat-file -e 1698b32033f38a034572891aed698609da2ff392:<path>
```

Model weights, user projects, voices and generated media are not part of the
Git distribution. See `THIRD_PARTY_NOTICES.md` for model and asset boundaries.
