"""Split each cleaned name pool into a train slice and an eval-holdout slice, so the
regression eval (eval/sa_names_eval_gen.py) measures recall on names the training corpus
never saw -- otherwise build_training_data.py and the eval generator would draw from the
same pool and the eval would partly be measuring memorization, not generalization.

Writes:
    data/pools/train/names_<group>.txt   (85%, used by build_training_data.py)
    data/pools/eval/names_<group>.txt    (15%, used by eval/sa_names_eval_gen.py)

Only splits the groups the eval set actually covers (see EVAL_GROUPS below, matching
eval/sa_names_eval_gen.py's existing NAMES groups). Siswati has training pools but isn't
part of the eval taxonomy, so it isn't split -- all of it stays available for training.

Usage:
    python scripts/corpus/split_eval_holdout.py
"""

import argparse
import random
from pathlib import Path

DATA_POOLS = Path(__file__).resolve().parents[2] / "data" / "pools"

EVAL_GROUPS = [
    "isiZulu",
    "isiXhosa",
    "Sesotho_Setswana",
    "Tshivenda_Xitsonga",
    "Afrikaans",
    "Western_control",
]
TRAIN_ONLY_GROUPS = ["Siswati"]

EVAL_HOLDOUT_RATIO = 0.15


def load_pool(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_HOLDOUT_RATIO)
    args = parser.parse_args()

    train_dir = DATA_POOLS / "train"
    eval_dir = DATA_POOLS / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: split each eval group's own pool into train/eval, per-group.
    group_train_names: dict[str, list[str]] = {}
    reserved_eval_names: set[str] = set()  # global, case-insensitive -- kept out of ALL training pools
    for group in EVAL_GROUPS:
        src = DATA_POOLS / f"names_{group}.txt"
        names = load_pool(src)
        if not names:
            print(f"[warn] no names pool for {group} at {src} -- run extract_pools.py first, skipping")
            continue
        rng = random.Random(args.seed)
        names = sorted(names)  # deterministic order before shuffling
        rng.shuffle(names)
        n_eval = max(1, int(len(names) * args.eval_ratio))
        eval_names, train_names = names[:n_eval], names[n_eval:]
        (eval_dir / f"names_{group}.txt").write_text("\n".join(sorted(eval_names)) + "\n")
        reserved_eval_names.update(n.lower() for n in eval_names)
        group_train_names[group] = train_names
        print(f"[{group}] {len(train_names)} train (pre-filter) / {len(eval_names)} eval-holdout")

    # Pass 2: same surname can be extracted from more than one corpus (e.g. "Mahlangu" is a
    # shared Nguni surname across isiZulu/Siswati pools) -- drop any name from every training
    # pool, including groups not in EVAL_GROUPS, that also landed in ANY eval-holdout slice.
    for group, train_names in group_train_names.items():
        filtered = [n for n in train_names if n.lower() not in reserved_eval_names]
        dropped = len(train_names) - len(filtered)
        (train_dir / f"names_{group}.txt").write_text("\n".join(sorted(filtered)) + "\n")
        print(f"[{group}] {len(filtered)} train (final, dropped {dropped} cross-group leaks)")

    for group in TRAIN_ONLY_GROUPS:
        src = DATA_POOLS / f"names_{group}.txt"
        if not src.exists():
            print(f"[warn] no names pool for {group} at {src} -- run extract_pools.py first, skipping")
            continue
        names = load_pool(src)
        filtered = [n for n in names if n.lower() not in reserved_eval_names]
        dropped = len(names) - len(filtered)
        (train_dir / f"names_{group}.txt").write_text("\n".join(sorted(filtered)) + "\n")
        print(f"[{group}] {len(filtered)} kept for training (dropped {dropped} cross-group leaks)")


if __name__ == "__main__":
    main()
