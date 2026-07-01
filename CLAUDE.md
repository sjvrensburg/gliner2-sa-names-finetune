# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A task scaffold for fine-tuning `SemplificaAI/gliner2-privacy-filter-PII-multi` (GLiNER2-PII,
the on-device PII model used by [privacy-ext](https://github.com/sjvrensburg/privacy-ext)) to
fix a measured recall gap on South African names in cue-free, label-style text (address/attn
lines). As of the initial scaffold, only docs, an eval generator/runner, and a vendored ONNX
export script exist — the training corpus and fine-tuning script itself are not yet built.

Read `README.md` first — it has the full problem statement, task outline, and repo layout.
The `docs/` files it links to contain the actual implementation guidance (data sources, JSONL
training format, LoRA config/hyperparameters, ONNX export invocation). Don't duplicate that
guidance from memory; re-read the relevant doc when working on that step, since the plan is
still evolving.

## Key constraints (don't relitigate these)

- Fine-tune **from** `SemplificaAI/gliner2-privacy-filter-PII-multi`, never from the bare
  `fastino/gliner2-base-v1` — starting from base loses the existing PII tuning.
- Use LoRA (`gliner2.training.trainer.GLiNER2Trainer` + `TrainingConfig(use_lora=True, ...)`),
  starting with `lora_target_modules=["encoder"]` only. Widen incrementally
  (`+span_rep` → `+classifier` → full default) only if eval shows it's needed — see
  `docs/TRAINING.md` for the widening order and rationale.
- Export only with `scripts/export_gliner2_onnx_fragments_v2.py` (vendored from
  `SemplificaAI/gliner2-rs`, Apache-2.0). This is the only exporter that produces the
  8-fragment ONNX split (`encoder`, `token_gather`, `span_rep`, `schema_gather`,
  `count_pred_argmax`, `count_lstm_fixed`, `scorer`, `classifier`) that privacy-ext's
  `gliner2-rs` Rust inference engine expects. Do not substitute other exporters (e.g.
  `lmoe/gliner2-onnx` — incompatible 4-fragment split).
- Training data label set must match privacy-ext's deployed schema (`name, street address,
  email, phone_num, id_num, url, username` — see `DEFAULT_LABELS` in privacy-ext's
  `server/src/lib.rs`). Training on a different label set won't transfer to the running server.
- Data sourcing: don't hand-pick or reuse a tiny name list — real-name diversity is the
  bottleneck for this task. Prefer nwu-ctext (CC-BY 2.5 SA) + MphayaNER (Apache 2.0) per
  `docs/DATA_SOURCES.md`; MasakhaNER 2.0 is CC-BY-NC (eval/research only, not for fine-tuning a
  shipped model).
- Weight synthetic training examples toward `address_attn`-style contexts (the measured
  failure) but keep the other 6 contexts well-represented too, to avoid regressing them.

## Common commands

Setup:
```sh
pip install gliner2 onnx onnxruntime huggingface_hub torch
```

Generate/run the regression eval (requires a running privacy-ext `pii-server` on
`127.0.0.1:8731`):
```sh
python eval/sa_names_eval_gen.py && python eval/run_sa_names_eval.py <bearer_token>
```
`sa_names_eval.jsonl` is generated, not checked in. The built-in ~40-name list is a first pass
only — extend it with a real name pool before using it as the fine-tuning gate.

Export a fine-tuned checkpoint to the 8-fragment ONNX format:
```sh
python scripts/export_gliner2_onnx_fragments_v2.py --model_path <checkpoint_or_hf_id> --out_dir <out_dir>
```

Validate a training JSONL file before training (see `docs/TRAINING_DATA_FORMAT.md`):
```python
from gliner2.training.data import TrainingDataset
dataset = TrainingDataset.load("train.jsonl")
dataset.validate(strict=True, raise_on_error=True)
```

## Repo layout

```
data/     training corpora (not yet present — generated locally or pulled from external sources)
eval/     266-case regression eval: generator (sa_names_eval_gen.py) + runner (run_sa_names_eval.py)
scripts/  vendored ONNX export script (export_gliner2_onnx_fragments_v2.py)
docs/     data sources, training data format, and LoRA training/export guidance
```

## Definition of done for a fine-tune iteration

Re-run the eval against both the currently-shipped fragments and the newly exported ones.
Success = address_attn recall rises substantially across the SA name groups (isiZulu,
isiXhosa, Sesotho, Setswana, Tshivenda, Xitsonga) with no material regression in the Western
control or the other 6 contexts (which already score 97-100%).
