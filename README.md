# gliner2-sa-names-finetune

Fine-tuning [GLiNER2-PII](https://huggingface.co/SemplificaAI/gliner2-privacy-filter-PII-multi)
(the on-device PII model used by [privacy-ext](https://github.com/sjvrensburg/privacy-ext)) to
close a measured detection gap for South African (isiZulu, isiXhosa, Sesotho, Setswana,
Tshivenda, Xitsonga) personal names.

## Background

privacy-ext is an on-device PII redactor: a local Rust daemon runs GLiNER2 (205M params, ONNX,
8 fragments) and a browser extension intercepts pastes to offer redacted text. It's a privacy
tool, so a name that goes undetected is a real leak, not just a lower score.

An eval of 266 labeled cases (6 name-origin groups × 7 sentence contexts) found:

- **Overall recall 95.9%** — not a broad failure.
- Every non-Western name group underperforms the Western control (100%): Sesotho/Setswana
  97.6%, isiZulu 96.4%, Tshivenda/Xitsonga 91.4%, isiXhosa 91.1%.
- **The gap is almost entirely concentrated in one context**: bare address/attn-line phrasing
  (`"Attn: {name}, Unit 4B, Soweto. Please call before delivery."`) scored only **73.7%**
  recall, while all 6 other contexts (explicit name-cue sentences, billing, form fields, email
  signatures, narrative, third-person) scored 97-100% across every group.
- Within `address_attn` alone: Tshivenda/Xitsonga 40%, isiXhosa 50%, isiZulu 75%.
- When a name IS detected, span boundaries are always exact (0 boundary errors) — this is a
  pure miss/no-miss recall problem, not a subword-fragmentation/boundary problem, despite SA
  names tokenizing into more subword pieces than Western names.

**The task: fine-tune GLiNER2-PII (LoRA) so it reliably detects SA names in short, cue-free,
label-style text (address lines, attn lines, delivery labels) — without regressing recall on
the contexts and name groups that already work.**

## Task outline

1. **Build the training corpus.**
   - Pull real SA name lists from liberally-licensed sources (see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)) —
     don't hand-pick names, and don't reuse a tiny ad-hoc list; recall on this task is
     bottlenecked on name-list size and diversity.
   - Weight generated examples toward `address_attn`-style contexts (the measured failure),
     but keep a healthy mix of the other 6 contexts too, so training doesn't overfit to one
     phrasing and forget the contexts that already work.
   - Target format is GLiNER2's training JSONL: `{"input": "...", "output": {"entities":
     {"name": ["..."]}}}`. See [`docs/TRAINING_DATA_FORMAT.md`](docs/TRAINING_DATA_FORMAT.md).
   - Include negative/other-entity examples too (addresses, emails, phone numbers) so the
     model doesn't start over-triggering on "name" for everything in an address block.

2. **Fine-tune with LoRA.**
   - Start from `fastino/gliner2-privacy-filter-PII-multi` (PyTorch, trainable) — the
     upstream checkpoint privacy-ext's ONNX fragments were exported from — not the bare
     `fastino/gliner2-base-v1`. Starting from the base model would lose the existing PII
     tuning. Note: `SemplificaAI/gliner2-privacy-filter-PII-multi` (what privacy-ext's Rust
     server downloads at runtime) only hosts the ONNX export fragments, not a trainable
     checkpoint — `GLiNER2.from_pretrained` on it 404s.
   - Use `gliner2.training.trainer.GLiNER2Trainer` + `TrainingConfig(use_lora=True, ...)`.
     LoRA checkpoints save already-merged weights, ready for export — no separate merge step.
   - Start with `lora_target_modules=["encoder"]` — the failure is pure recall-without-cue,
     not span-boundary or classification, so encoder-only adaptation is the surgical starting
     point. Only widen to `+span_rep`/`+classifier` if that's insufficient.
   - See [`docs/TRAINING.md`](docs/TRAINING.md) for a concrete config and hardware notes.

3. **Export back to the 8-fragment ONNX format.**
   - privacy-ext's Rust server (`gliner2-rs`) expects a specific 8-fragment ONNX split, not a
     single graph. Use `scripts/export_gliner2_onnx_fragments_v2.py` (vendored here from
     `SemplificaAI/gliner2-rs`, Apache-2.0) — do not reach for third-party exporters like
     `lmoe/gliner2-onnx`; that project uses an incompatible 4-fragment split and will not work
     with privacy-ext's inference engine.
   - `python scripts/export_gliner2_onnx_fragments_v2.py --model_path <checkpoint> --out_dir <dir>`

4. **Re-run the regression eval before calling it done.**
   - `eval/` contains the eval-set generator and runner used to produce the numbers above.
     Regenerate with a larger name list, then run it against both the old and new fragments
     and compare — the goal is address_attn recall going up on every SA group with no material
     regression elsewhere (Western control, or the 6 already-good contexts).

## Repo layout

```
data/     training corpora (generated here, or pulled from external sources — see DATA_SOURCES.md)
eval/     the 266-case eval set + runner, and the regression eval to build on
scripts/  vendored ONNX export script + any training-launch scripts
docs/     detailed guidance referenced above
```

## Non-goals

- This is not a general SA-language NLP project — scope is narrowly "make GLiNER2-PII detect
  SA names reliably across contexts," not building a general SA NER dataset or model.
- Not attempting to fix every possible context/name combination in one pass — the address_attn
  gap is the measured, prioritized target.
