# Third-party notices and model boundaries

This document records bundled third-party source and separately downloaded
model dependencies. It is an attribution and scope record, not a grant of any
additional right. The exact path rules are in `LICENSES/README.md`.

## Bundled source and browser assets

| Component | Repository path | Pinned source or version | License retained in this repository |
| --- | --- | --- | --- |
| IndexTTS2 | Upstream-derived paths identified in `LICENSES/README.md` | `index-tts/index-tts` commit `1698b32033f38a034572891aed698609da2ff392` | bilibili IndexTTS2 Model Use License: `LICENSES/Bilibili-IndexTTS.txt`, `LICENSES/Bilibili-IndexTTS-ZH.txt` |
| CosyVoice | `cosyvoice/**` excluding Matcha-TTS | `FunAudioLLM/CosyVoice` commit `c93d3dda01ae69fde8a4b2372f0e4260135599c3` plus the local adaptations recorded in `docs/third_party_sources.md` | Apache-2.0: `cosyvoice/LICENSE` |
| Matcha-TTS | `cosyvoice/third_party/Matcha-TTS/**` | `shivammehta25/Matcha-TTS` commit `dd9105b34bf2be2230f4aa1e4769fb586a3c824e` | MIT: adjacent `LICENSE` |
| Matcha-TTS HiFi-GAN material | `cosyvoice/third_party/Matcha-TTS/matcha/hifigan/**` | Vendored with Matcha-TTS | MIT: adjacent `LICENSE` |
| Tailwind CSS | `dubbing_service/webui/vendor/tailwindcss-3.4.17/**` | 3.4.17 | MIT: adjacent `LICENSE` |
| Vue | `dubbing_service/webui/vendor/vue-3.5.39/**` | 3.5.39 | MIT: adjacent `LICENSE` |
| WaveSurfer | `dubbing_service/webui/vendor/wavesurfer-7.12.10/**` | 7.12.10 | BSD-3-Clause: adjacent `LICENSE` |

Browser asset hashes and acquisition details are recorded in
`dubbing_service/webui/vendor/VERSIONS.md`. Package names in `uv.lock`, the two
runtime lockfiles and `package-lock.json` remain subject to their respective
upstream licenses; a lockfile does not relicense a dependency.

## Mandatory IndexTTS2 derivative notice

The retained IndexTTS2 license requires the following exact statement on a
distribution page or in accompanying documentation for a derivative work:

> Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.

The Chinese license expresses the corresponding obligation as:

> 该衍生品对原模型所作的任何改动与原模型原始权利人无关，原始权利人对该衍生品不背书、不担保、不承担责任。

Keep both `LICENSES/Bilibili-IndexTTS.txt` and
`LICENSES/Bilibili-IndexTTS-ZH.txt` with every distributed copy. The license
states that the Chinese text prevails if the language versions conflict. Its
scale threshold is not identical in the two language versions:
the Chinese text refers to annual revenue above RMB 100 million, while the
English text refers to RMB 1 billion. Do not resolve that discrepancy by
silently choosing the less restrictive number; obtain the required separate
license or professional advice when the threshold may be relevant.

## Model weights downloaded separately

Model weights are intentionally absent from Git. The manifests pin the source,
revision, file size and SHA-256, but manifest metadata grants no model right.
Users download and accept the applicable model terms themselves.

| Model source | Manifest | Recorded license boundary |
| --- | --- | --- |
| `IndexTeam/IndexTTS-2` | `models/manifests/index_tts2.json` | IndexTTS model license; review the source model license before redistribution or commercial use |
| `AI-ModelScope/w2v-bert-2.0` | `models/manifests/index_tts2.json` | MIT |
| `amphion/MaskGCT` | `models/manifests/index_tts2.json` | **CC-BY-NC-4.0; non-commercial only unless separate permission applies** |
| `iic/speech_campplus_sv_zh-cn_16k-common` | `models/manifests/index_tts2.json` | ModelScope model license; review the model page terms |
| `nv-community/bigvgan_v2_22khz_80band_256x` | `models/manifests/index_tts2.json` | MIT |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `models/manifests/qwen3_tts_customvoice.json` | Apache-2.0 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `models/manifests/qwen3_tts_voicedesign.json` | Apache-2.0 |
| `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` and `pengzhendong/wetext` | `models/manifests/cosyvoice3.json` | Apache-2.0 as recorded by the manifest |
| Whisper / faster-whisper model sources | `models/manifests/whisper_*.json` | MIT as recorded by each manifest |
| `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | `models/manifests/whisperx_zh_alignment.json` | Apache-2.0 |

### MaskGCT commercial red line

The current IndexTTS2 supply manifest includes MaskGCT material marked
CC-BY-NC-4.0. A technically successful IndexTTS2 run is therefore **not** proof
that the resulting pipeline is cleared for commercial production. Before any
revenue-generating or other commercial use, replace that dependency with a
commercially compatible component, obtain separate permission, or obtain a
qualified legal determination. Apache-2.0 coverage of the platform code does
not remove this restriction.

## Voice, media and output rights

The upstream `examples/**`, `tests/sample_prompt.wav` and `assets/**` are kept
as upstream IndexTTS materials. Their presence is not a separate representation
by TadashiJLU that a speaker identity, recording, image, video or generated
output is cleared for commercial reuse. Users remain responsible for voice,
performance, copyright, privacy, personality-right, music, sound-effect,
cover-art and synthetic-content disclosure obligations.
