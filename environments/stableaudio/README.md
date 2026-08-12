# Stable Audio 3 Small-SFX runtime

This isolated Python 3.11 environment runs the headless local SFX worker. It is
kept separate because Stable Audio 3 pins PyTorch 2.7.1 and requires a newer
Transformers release than the platform runtime.

The upstream source is pinned to commit
`a0b57f5483c4588f827f3552b7d5c6ca2a9687be`. This lock is intentionally
Windows-only and resolves the PyTorch 2.7.1 CUDA 12.8 wheels from the explicit
PyTorch index. Linux deployment remains outside the project's validated
runtime matrix and must use its own reviewed lock.

The worker never downloads a model. Before loading, place a materialized copy
of the gated model below:

```text
models/stable_audio_3/small-sfx/
  model_config.json
  model.safetensors
  t5gemma-b-b-ul2/
    config.json
    generation_config.json
    model.safetensors
    special_tokens_map.json
    tokenizer.json
    tokenizer.model
    tokenizer_config.json
```

Access to the upstream model requires accepting the Stability AI and Gemma
terms. Provisioning and manifest verification are intentionally handled by the
platform model-supply workflow, not by this runtime.

Create or repair this environment through the repository bootstrap workflow.
Normal startup must call this environment's interpreter directly and must not
perform an implicit dependency sync.
