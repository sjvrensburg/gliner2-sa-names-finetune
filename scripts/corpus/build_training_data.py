"""Generate the SA names + multi-entity training corpus (data/train_raw.jsonl).

Extends eval/sa_names_eval_gen.py's NAMES/CONTEXTS dict pattern: templates are format
strings with named placeholders (`{name}`, `{street_address}`, `{phone}`, `{email}`, ...).
Spans are computed while building the string (not via str.find afterwards), so repeated
substrings never cause offset collisions.

Usage:
    python scripts/corpus/build_training_data.py --target-size 2500
"""

import argparse
import json
import random
import string as _string
from pathlib import Path

from synth import fake_email, fake_id_num, fake_phone, fake_street_address, fake_url, fake_username

DATA_POOLS = Path(__file__).resolve().parents[2] / "data" / "pools"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

GROUPS = [
    "isiZulu",
    "isiXhosa",
    "Sesotho_Setswana",
    "Tshivenda_Xitsonga",
    "Afrikaans",
    "Siswati",
    "Western_control",
]

# label -> template placeholder name
ENTITY_LABELS = {
    "name": "name",
    "street address": "street_address",
    "email": "email",
    "phone_num": "phone",
    "id_num": "id_num",
    "url": "url",
    "username": "username",
}

# Single-entity templates (name only) -- weight kept low relative to address-style ones,
# per docs/TRAINING_DATA_FORMAT.md.
SINGLE_ENTITY_TEMPLATES = {
    "cue_explicit": ("Hi, my name is {name} and I'd like to schedule an appointment for next week.", 8),
    "billing": ("Invoice #4471 billed to {name} - payment overdue.", 8),
    "form_field": ("Full Name: {name}", 8),
    "email_sig": ("Kind regards,\n{name}\nSenior Consultant", 6),
    "narrative_no_cue": ("{name} arrived at the office around 9am and left the documents at reception.", 8),
    "third_person": ("The parcel was collected by {name} on Tuesday afternoon.", 8),
}

# Address/label-style templates -- the measured failure mode, weighted heavily.
ADDRESS_TEMPLATES = {
    "address_attn": ("Attn: {name}, Unit 4B, {street_address}. Please call before delivery.", 20),
    "address_attn_2": ("Deliver to: {name}\n{street_address}", 18),
    "label_field": ("{name}\nAccount Number: 88213-04", 14),
    "shipping_label": ("SHIP TO\n{name}\n{street_address}", 18),
}

# Multi-entity templates combining name with other PII types so the model learns not to
# over-trigger `name` on every token in an address/contact block.
MULTI_ENTITY_TEMPLATES = {
    "address_full_block": (
        "SHIP TO\n{name}\n{street_address}\nTel: {phone}", 12
    ),
    "contact_card": (
        "{name}\n{email}\n{phone}", 10
    ),
    "form_with_contact": (
        "Full Name: {name}\nEmail: {email}\nPhone: {phone}\nID Number: {id_num}", 8
    ),
    "profile_block": (
        "{name}\nUsername: {username}\nProfile: {url}", 6
    ),
}

ALL_TEMPLATES = {**SINGLE_ENTITY_TEMPLATES, **ADDRESS_TEMPLATES, **MULTI_ENTITY_TEMPLATES}


def load_pool(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_placeholder_values(name: str, place_pool: list[str]) -> dict[str, str]:
    return {
        "name": name,
        "street_address": fake_street_address(place_pool),
        "phone": fake_phone(),
        "email": fake_email(name),
        "id_num": fake_id_num(),
        "url": fake_url(),
        "username": fake_username(name),
    }


PLACEHOLDER_TO_LABEL = {v: k for k, v in ENTITY_LABELS.items()}


def render_with_spans(template: str, values: dict[str, str]) -> tuple[str, dict[str, list[str]]]:
    """Render `template` and return (text, entities) with each entity value's exact
    substring recorded -- built incrementally so offsets never collide with repeats."""
    formatter = _string.Formatter()
    text_parts = []
    entities: dict[str, list[str]] = {}
    cursor = 0
    for literal, field_name, _fmt, _conv in formatter.parse(template):
        text_parts.append(literal)
        cursor += len(literal)
        if field_name is None:
            continue
        value = values[field_name]
        text_parts.append(value)
        label = PLACEHOLDER_TO_LABEL[field_name]
        entities.setdefault(label, []).append(value)
        cursor += len(value)
    return "".join(text_parts), entities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(DATA_DIR / "train_raw.jsonl"))
    args = parser.parse_args()
    random.seed(args.seed)

    template_keys = list(ALL_TEMPLATES.keys())
    weights = [ALL_TEMPLATES[k][1] for k in template_keys]

    group_pools = {}
    for group in GROUPS:
        names = load_pool(DATA_POOLS / f"names_{group}.txt")
        places = load_pool(DATA_POOLS / f"places_{group}.txt")
        if not names:
            print(f"[warn] no names pool for {group} -- run extract_pools.py first, skipping group")
            continue
        group_pools[group] = (names, places)

    if not group_pools:
        raise SystemExit("no name pools found under data/pools/ -- run fetch_sources.py + extract_pools.py first")

    rows = []
    active_groups = list(group_pools.keys())
    for i in range(args.target_size):
        group = active_groups[i % len(active_groups)]
        names, places = group_pools[group]
        name = random.choice(names)
        template_key = random.choices(template_keys, weights=weights, k=1)[0]
        template, _ = ALL_TEMPLATES[template_key]
        values = build_placeholder_values(name, places)
        text, entities = render_with_spans(template, values)
        rows.append({"input": text, "output": {"entities": entities}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
