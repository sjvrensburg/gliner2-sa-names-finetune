from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

model = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi")

config = TrainingConfig(
    output_dir="./sa_names_lora",
    experiment_name="sa_names_address_context",

    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["encoder"],

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
