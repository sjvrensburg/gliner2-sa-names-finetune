"""Fetch raw NER corpora used to build the SA names training corpus.

Downloads (or reuses a local cache of):
  - nwu-ctext / NCHLT NER corpora (isiZulu, isiXhosa, Sesotho, Setswana, Siswati),
    CC-BY 2.5 South Africa. Their Hugging Face dataset repos (`nwu-ctext/<lang>_ner_corpus`)
    only ship a legacy `datasets` loading script (no longer supported by the `datasets`
    library) whose `_URL` points at a SADiLaR bitstream zip -- so these are downloaded
    directly from that same SADiLaR URL and extracted to a CoNLL-format .txt file, exactly
    reproducing what the (now-broken) loading script used to do.
  - Afrikaans NER corpus, which HF hosts as a plain parquet file (no script), so it loads
    normally via `datasets.load_dataset`.
  - MphayaNER (Tshivenda) via git clone from GitHub, Apache-2.0.
  - WikiANN English config (Western_control name/place pool), via `datasets`, CC BY-SA
    (Wikipedia-derived).

Everything is written under data/raw/ (gitignored) and re-fetched from scratch each run
unless --force is omitted and the target already exists, matching eval/'s
regenerate-don't-commit pattern. See docs/DATA_SOURCES.md for license details.
"""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

import requests

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# lang -> (SADiLaR bitstream zip URL, path to the CoNLL .txt inside the zip)
SADILAR_SOURCES = {
    "isiZulu": (
        "https://repo.sadilar.org/bitstream/handle/20.500.12185/319/nchlt_isizulu_named_entity_annotated_corpus.zip?sequence=3&isAllowed=y",
        "NCHLT isiZulu Named Entity Annotated Corpus/Dataset.NCHLT-II.zu.NER.Full.txt",
    ),
    "isiXhosa": (
        "https://repo.sadilar.org/bitstream/handle/20.500.12185/312/nchlt_isixhosa_named_entity_annotated_corpus.zip?sequence=3&isAllowed=y",
        "NCHLT isiXhosa Named Entity Annotated Corpus/Dataset.NCHLT-II.xh.NER.Full.txt",
    ),
    "Sesotho": (
        "https://repo.sadilar.org/bitstream/handle/20.500.12185/334/nchlt_sesotho_named_entity_annotated_corpus.zip?sequence=3&isAllowed=y",
        "NCHLT Sesotho Named Entity Annotated Corpus/Dataset.NCHLT-II.st.NER.Full.txt",
    ),
    "Setswana": (
        "https://repo.sadilar.org/bitstream/handle/20.500.12185/341/nchlt_setswana_named_entity_annotated_corpus.zip?sequence=3&isAllowed=y",
        "NCHLT Setswana Named Entity Annotated Corpus/Dataset.NCHLT-II.tn.NER.Full.txt",
    ),
    "Siswati": (
        "https://repo.sadilar.org/bitstream/handle/20.500.12185/346/nchlt_siswati_named_entity_annotated_corpus.zip?sequence=3&isAllowed=y",
        "NCHLT Siswati Named Entity Annotated Corpus/Dataset.NCHLT-II.ss.NER.Full.txt",
    ),
}

AFRIKAANS_DATASET = "nwu-ctext/afrikaans_ner_corpus"
MPHAYANER_REPO = "https://github.com/rendanim/MphayaNER.git"
WIKIANN_DATASET = "unimelb-nlp/wikiann"
WIKIANN_CONFIG = "en"


def fetch_sadilar_corpora(force: bool) -> None:
    for lang, (url, extracted_file) in SADILAR_SOURCES.items():
        out_dir = DATA_RAW / "nwu-ctext" / lang
        out_txt = out_dir / "corpus.txt"
        if out_txt.exists() and not force:
            print(f"[skip] {lang} already cached at {out_txt}")
            continue
        print(f"[fetch] {lang} <- {url}")
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / "_download.zip"
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(extracted_file) as src, open(out_txt, "wb") as dst:
                shutil.copyfileobj(src, dst)
        zip_path.unlink()


def fetch_afrikaans(force: bool) -> None:
    from datasets import load_dataset

    out_dir = DATA_RAW / "nwu-ctext" / "Afrikaans"
    if out_dir.exists() and not force:
        print(f"[skip] Afrikaans already cached at {out_dir}")
        return
    print(f"[fetch] {AFRIKAANS_DATASET} -> {out_dir}")
    ds = load_dataset(AFRIKAANS_DATASET)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))


def fetch_mphayaner(force: bool) -> None:
    out_dir = DATA_RAW / "mphayaner"
    if out_dir.exists():
        if not force:
            print(f"[skip] MphayaNER already cached at {out_dir}")
            return
        subprocess.run(["rm", "-rf", str(out_dir)], check=True)
    print(f"[fetch] {MPHAYANER_REPO} -> {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", MPHAYANER_REPO, str(out_dir)], check=True
    )


def fetch_wikiann_en(force: bool) -> None:
    from datasets import load_dataset

    out_dir = DATA_RAW / "wikiann-en"
    if out_dir.exists() and not force:
        print(f"[skip] WikiANN-en already cached at {out_dir}")
        return
    print(f"[fetch] {WIKIANN_DATASET} ({WIKIANN_CONFIG}) -> {out_dir}")
    ds = load_dataset(WIKIANN_DATASET, WIKIANN_CONFIG)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch even if already cached"
    )
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    fetch_sadilar_corpora(args.force)
    fetch_afrikaans(args.force)
    fetch_mphayaner(args.force)
    fetch_wikiann_en(args.force)
    print("done.")


if __name__ == "__main__":
    main()
