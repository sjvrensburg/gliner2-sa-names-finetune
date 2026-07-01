# Fine-tuning guide

## Setup

```sh
pip install gliner2 onnx onnxruntime huggingface_hub torch
```

## Hardware notes

Verified feasible on a modest laptop GPU: RTX A2000 8GB, 20 CPU threads, 31GB RAM. LoRA
fine-tuning of GLiNER2 (205M params) fits comfortably in 8GB VRAM at `batch_size=16` with
`fp16=True`. Full (non-LoRA) fine-tuning is also possible with gradient checkpointing + a
smaller batch, but LoRA is the recommended default here — lower VRAM, and much lower risk of
catastrophic forgetting on the other 41 PII entity types the base checkpoint already handles.

If no GPU is available, CPU-only training works too (just much slower) — no code changes
needed, `torch`/`transformers` fall back automatically.

## Minimal training script

```python
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Fine-tune FROM the deployed checkpoint, not the bare base model —
# otherwise you lose the existing PII-specific tuning.
model = GLiNER2.from_pretrained("SemplificaAI/gliner2-privacy-filter-PII-multi")

config = TrainingConfig(
    output_dir="./sa_names_lora",
    experiment_name="sa_names_address_context",

    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["encoder"],  # start narrow: the failure is a recall
                                       # problem, not span-boundary/classification

    batch_size=16,
    task_lr=5e-4,
    num_epochs=15,
    fp16=True,

    eval_strategy="epoch",
    save_best=True,
    early_stopping=True,
    early_stopping_patience=3,
)

trainer = GLiNER2Trainer(model, config)
results = trainer.train(train_data="data/train.jsonl", eval_data="data/val.jsonl")
print(results)

# LoRA checkpoints are already merged — ready to export directly.
best_model_path = "./sa_names_lora/best"
```

## If encoder-only LoRA isn't enough

Widen `lora_target_modules` incrementally rather than jumping straight to "all modules":

1. `["encoder"]` (starting point)
2. `["encoder", "span_rep"]`
3. `["encoder", "span_rep", "classifier"]`
4. Full default: `["encoder", "span_rep", "classifier", "count_embed", "count_pred"]`

Re-run the eval after each widening — don't skip straight to maximum adaptation, since more
adapted modules = more forgetting risk on the entity types/contexts that already work.

## Export to ONNX

```sh
python scripts/export_gliner2_onnx_fragments_v2.py \
  --model_path ./sa_names_lora/best \
  --out_dir ./onnx_out
```

This produces the 8-fragment set (`encoder`, `token_gather`, `span_rep`, `schema_gather`,
`count_pred_argmax`, `count_lstm_fixed`, `scorer`, `classifier`) in fp32/fp16/fp16_iobinding
variants, matching what privacy-ext's `gliner2-rs` engine expects — this is the exact script
used to produce the fragments privacy-ext already ships.

## Regression-testing before shipping

Point privacy-ext's `pii-server` at the new fragments (`PII_MODELS_DIR=./onnx_out`) and re-run
the eval harness from `eval/`. Compare against the baseline numbers in this repo's README:
address_attn recall should rise substantially for isiZulu/isiXhosa/Sesotho/Setswana/
Tshivenda/Xitsonga, with no material drop in the Western control or the other 6 contexts.
