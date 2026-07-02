import json
import random
from pathlib import Path

# Fallback list used only if data/pools/eval/ (built by scripts/corpus/split_eval_holdout.py)
# isn't present -- this was the original ~40-name first pass, kept as a bootstrap so this
# script still runs standalone before the corpus pipeline has ever been executed.
FALLBACK_NAMES = {
    "isiZulu": ["Nomvula Mahlangu", "Thabo Nkosi", "Sipho Zungu", "Bongani Ndlovu",
                "Lindiwe Khumalo", "Nokuthula Mkhize", "Mandla Buthelezi", "Zanele Cele"],
    "isiXhosa": ["Sibusiso Ngcobo", "Ayanda Xulu", "Anele Mgcina", "Lithemba Sonjica",
                 "Nomalanga Gqamana", "Buhle Mtshali", "Zukisa Dyantyi", "Loyiso Nqevu"],
    "Sesotho_Setswana": ["Palesa Mokoena", "Katlego Molefe", "Tshepo Rakgoale",
                         "Lerato Sekgobela", "Kabelo Mothibi", "Refilwe Mmusi"],
    "Tshivenda_Xitsonga": ["Rendani Mulaudzi", "Vhonani Netshitenzhe", "Hlengiwe Chauke",
                           "Ndivhuwo Ramulondi", "Rirhandzu Baloyi"],
    "Afrikaans": ["Pieter van der Merwe", "Annemarie Botha", "Willem Kruger",
                  "Susanna Fourie", "Hendrik Nieuwoudt"],
    "Western_control": ["John Smith", "Maria Garcia", "David Johnson", "Emma Wilson",
                        "Wei Chen", "Fatima Al-Sayed"],
}

# Cap names sampled per group so total case count (names x contexts) stays a manageable
# eval run, even though the held-out pools can have hundreds of real names per group.
NAMES_PER_GROUP = 40
EVAL_POOLS_DIR = Path(__file__).resolve().parent.parent / "data" / "pools" / "eval"


def load_names() -> dict[str, list[str]]:
    names = {}
    rng = random.Random(42)
    for group in FALLBACK_NAMES:
        pool_path = EVAL_POOLS_DIR / f"names_{group}.txt"
        if pool_path.exists():
            pool = [line.strip() for line in pool_path.read_text().splitlines() if line.strip()]
            if pool:
                sample = pool if len(pool) <= NAMES_PER_GROUP else rng.sample(pool, NAMES_PER_GROUP)
                names[group] = sorted(sample)
                continue
        print(f"[warn] no eval-holdout pool for {group} at {pool_path} -- "
              f"falling back to the small hardcoded list; run scripts/corpus/fetch_sources.py "
              f"+ extract_pools.py + split_eval_holdout.py for the real held-out eval set")
        names[group] = FALLBACK_NAMES[group]
    return names


NAMES = load_names()

CONTEXTS = {
    "cue_explicit":    "Hi, my name is {name} and I'd like to schedule an appointment for next week.",
    "billing":         "Invoice #4471 billed to {name} - payment overdue.",
    "address_attn":    "Attn: {name}, Unit 4B, Soweto. Please call before delivery.",
    "form_field":      "Full Name: {name}",
    "email_sig":       "Kind regards,\n{name}\nSenior Consultant",
    "narrative_no_cue":"{name} arrived at the office around 9am and left the documents at reception.",
    "third_person":    "The parcel was collected by {name} on Tuesday afternoon.",
}

rows = []
i = 0
for group, names in NAMES.items():
    for name in names:
        for ctx_key, template in CONTEXTS.items():
            text = template.format(name=name)
            start = text.find(name)
            rows.append({
                "id": i, "group": group, "context": ctx_key,
                "name": name, "text": text,
                "expected_start": start, "expected_end": start + len(name),
            })
            i += 1

from pathlib import Path

with open(Path(__file__).parent / "sa_names_eval.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} cases")
