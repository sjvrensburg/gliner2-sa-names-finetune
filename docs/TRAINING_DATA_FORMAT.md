# Training data format

GLiNER2 training data is JSONL, one example per line:

```jsonl
{"input": "Attn: Nomvula Mahlangu, Unit 4B, Soweto. Please call before delivery.", "output": {"entities": {"name": ["Nomvula Mahlangu"]}}}
```

Alternative equivalent key names: `{"text": ..., "schema": {"entities": {...}}}`.

## Requirements

- Each example needs at least one task (here: `entities`). Empty-everything examples are
  invalid and will fail validation.
- Entity mention strings must actually appear in `input` (checked under strict validation).
- Multiple entity types can co-occur — e.g. also label `street address`, `phone_num` etc. if
  present, matching the labels privacy-ext's server already uses
  (`name, street address, email, phone_num, id_num, url, username`, see
  `DEFAULT_LABELS` in privacy-ext's `server/src/lib.rs`). Keeping label names consistent with
  the deployed schema matters — training on a different label set won't transfer to the
  running server's classify calls.

## What to generate

Cross a name pool (from `DATA_SOURCES.md`) against context templates, weighted toward the
measured failure context. A reasonable starting template set (extend `eval/`'s existing
generator rather than starting from scratch):

```python
CONTEXTS = {
    "address_attn":     "Attn: {name}, Unit 4B, Soweto. Please call before delivery.",
    "address_attn_2":   "Deliver to: {name}\n12 Church Street, Durban 4001",
    "label_field":       "{name}\nAccount Number: 88213-04",
    "shipping_label":   "SHIP TO\n{name}\n45 Long Street",
    # keep a healthy proportion of the already-working contexts too, so training
    # doesn't overfit narrowly and regress them:
    "cue_explicit":     "Hi, my name is {name} and I'd like to schedule an appointment.",
    "billing":          "Invoice #4471 billed to {name} - payment overdue.",
    "form_field":       "Full Name: {name}",
    "email_sig":        "Kind regards,\n{name}\nSenior Consultant",
    "narrative_no_cue": "{name} arrived at the office around 9am.",
    "third_person":     "The parcel was collected by {name} on Tuesday afternoon.",
}
```

Aim for enough volume that each (name group × context) cell has multiple examples — the
original GLiNER2-PII model was trained on ~4,910 examples total, so a corpus in the
low-thousands is a reasonable target, not tens of thousands.

## Validation

Use GLiNER2's own dataset tooling to catch errors before training:

```python
from gliner2.training.data import TrainingDataset

dataset = TrainingDataset.load("train.jsonl")
dataset.validate(strict=True, raise_on_error=True)
dataset.print_stats()

train_data, val_data, _ = dataset.split(train_ratio=0.85, val_ratio=0.15, test_ratio=0.0, seed=42)
train_data.save("train.jsonl")
val_data.save("val.jsonl")
```
