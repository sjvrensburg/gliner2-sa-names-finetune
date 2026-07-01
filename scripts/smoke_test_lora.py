"""Quick smoke test: LoRA fine-tune on a tiny subset of data/train.jsonl, just to confirm
the training pipeline (model load -> LoRA -> train -> save) works end to end. Not a real
training run -- few samples, few steps, meant to finish in well under a minute on GPU.

Usage:
    python scripts/smoke_test_lora.py
"""

import time
from pathlib import Path

import torch
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

t0 = time.time()
print("loading base model...")
# Trainable PyTorch checkpoint -- NOT SemplificaAI/gliner2-privacy-filter-PII-multi, which
# only hosts ONNX export fragments (no config.json / weights, from_pretrained 404s there).
model = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")
print(f"loaded in {time.time() - t0:.1f}s")

config = TrainingConfig(
    output_dir="/tmp/smoke_test_lora_out",
    experiment_name="sa_names_smoke_test",
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["encoder"],
    batch_size=4,
    num_epochs=1,
    max_steps=10,
    max_train_samples=40,
    max_eval_samples=10,
    task_lr=5e-4,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=10,
    logging_steps=1,
    save_best=False,
    num_workers=0,
)

trainer = GLiNER2Trainer(model, config)
t1 = time.time()
results = trainer.train(
    train_data=str(DATA_DIR / "train.jsonl"),
    eval_data=str(DATA_DIR / "val.jsonl"),
)
print(f"train() finished in {time.time() - t1:.1f}s")
print("results:", results)
print(f"total: {time.time() - t0:.1f}s")
