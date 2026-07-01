"""Validate the generated corpus and split into train/val, per docs/TRAINING_DATA_FORMAT.md.

Usage:
    python scripts/corpus/validate_and_split.py
"""

import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    from gliner2.training.data import TrainingDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA_DIR / "train_raw.jsonl"))
    parser.add_argument("--train-out", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--val-out", default=str(DATA_DIR / "val.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = TrainingDataset.load(args.input)
    dataset.validate(raise_on_error=True)
    dataset.print_stats()

    train_data, val_data, _ = dataset.split(
        train_ratio=0.85, val_ratio=0.15, test_ratio=0.0, seed=args.seed
    )
    train_data.save(args.train_out)
    val_data.save(args.val_out)
    print(f"wrote {args.train_out} and {args.val_out}")


if __name__ == "__main__":
    main()
