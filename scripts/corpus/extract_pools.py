"""Extract flat name/place pools from the raw NER corpora fetched by fetch_sources.py.

For each name group, joins contiguous PERSON-tagged token spans into name strings and
contiguous LOCATION-tagged token spans into place strings, writing:
    data/pools/names_<group>.txt
    data/pools/places_<group>.txt

Groups and sources:
    isiZulu, isiXhosa, Sesotho_Setswana*, Siswati
        -> SADiLaR/NCHLT NER corpora, downloaded as CoNLL .txt by fetch_sources.py
           (`TOKEN\tTAG` per line, tags include B-PERS/I-PERS, B-LOC/I-LOC)
    Afrikaans
        -> nwu-ctext (HF `datasets` parquet, ClassLabel ner_tags: B-PERS/I-PERS, B-LOC/I-LOC)
    Tshivenda_Xitsonga
        -> MphayaNER (CoNLL file, `TOKEN TAG` per line, PER/LOC/ORG/DATE)
    Western_control
        -> WikiANN English config (IOB2 ner_tags: B-PER/I-PER, B-LOC/I-LOC)

PER/LOC tag prefixes are auto-detected per source (PERS vs PER) rather than hardcoded.

*Sesotho and Setswana are pooled together (matching the eval set's existing
Sesotho_Setswana group); Tshivenda (MphayaNER) stands in for the Tshivenda_Xitsonga group
since no Xitsonga corpus was identified in docs/DATA_SOURCES.md.

All 7 groups run through the same extraction logic below -- no hand-curated lists.
"""

import argparse
import re
from pathlib import Path

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_POOLS = Path(__file__).resolve().parents[2] / "data" / "pools"

MIN_TOKEN_LEN = 2  # drop single-character junk tokens
MAX_WORDS = 5  # entries longer than this are almost always titles/clauses, not names

# Bantu name-class concords the SADiLaR/MphayaNER annotators frequently left glued to the
# following proper noun (e.g. "uSteve Biko", "noKwitshana", "kwaMahlangu"). Stripped only
# from the NAMES pools -- places legitimately keep these (e.g. "eThekwini", "KwaZulu").
NAME_PREFIXES = [
    "kwa", "ngu", "nga", "kga", "kah", "wa", "ka", "ku", "no", "lu",
    "se", "ye", "ba", "na", "e", "o", "u",
]
_NAME_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(sorted(NAME_PREFIXES, key=len, reverse=True)) + r")(?=[A-Z])"
)

# Titles/honorifics annotators included inside PERSON spans -- strip as leading tokens.
HONORIFICS = {
    "mr", "mrs", "ms", "dr", "prof", "adv", "rev", "cllr", "hon", "chief", "kgosi",
    "nkosi", "mnu", "mme", "ngaka", "gr", "ugqr", "kgs", "mnr", "nkosikazi",
    "nkosaz", "inkhosi", "inkosi", "king", "queen", "kgosigadi",
}

# Job-title / office words (English, Afrikaans, isiXhosa/isiZulu, Sesotho/Setswana) that
# show up glued onto PERSON spans in these corpora. If ANY word in an entry matches one of
# these, the whole entry is dropped rather than truncated -- safer than guessing where the
# title ends and the actual name begins.
TITLE_WORDS = {
    "president", "premier", "archbishop", "minister", "secretary", "sekretaris",
    "commissioner", "kommissaris", "direkteur", "direkteur-generaal", "bestuurder",
    "uitvoerende", "hoofuitvoerende", "rektor", "kampusrektor", "registrateur",
    "inspekteur", "hoofinspekteur", "koordineerder", "koördineerder", "somlomo",
    "moporof", "mongameli", "bawo", "baw", "speaker", "judge", "justice", "regter",
    "advokaat", "governor", "goewerneur", "mayor", "burgemeester", "councillor",
    "raadslid", "molaodi", "molaodimogolo", "premiersvrou", "nasionale",
    "provinsiale",
}

# Lowercase connector words allowed inside an otherwise-Titlecase name (surname particles,
# royal-name "of", etc.) -- anything else lowercase inside a name/place is treated as a
# sign of a mis-tagged title/clause/species-name rather than a real proper noun.
LOWERCASE_PARTICLES = {
    "van", "der", "de", "la", "du", "von", "bin", "ibn", "al", "mac", "of", "the",
    "ka", "wa", "sa",
}


def _strip_prefix(word: str) -> str:
    return _NAME_PREFIX_RE.sub("", word, count=1)


def _strip_honorifics(tokens: list[str]) -> list[str]:
    while tokens and tokens[0].strip(".").lower() in HONORIFICS:
        tokens = tokens[1:]
    return tokens


def _looks_like_proper_noun(entry: str, require_leading_upper: bool) -> bool:
    """Reject entries that look like titles/clauses/species-names rather than proper
    nouns. For names (prefixes already stripped), each word must start uppercase. For
    places, a leading lowercase locative prefix glued to a capitalized stem is legitimate
    Nguni/Sotho orthography (e.g. "eThekwini", "kwaZulu"), so any uppercase letter in the
    word is accepted there instead of requiring one at position 0."""
    if any(ch.isdigit() for ch in entry):
        return False
    if any(ch in entry for ch in ":;|@/\\"):
        return False
    words = entry.split()
    if not words or len(words) > MAX_WORDS:
        return False
    if len(words) == 1 and words[0].isupper() and len(words[0]) <= 3:
        return False  # short all-caps acronym (e.g. "TV", "DCJ"), not a name
    for w in words:
        if w.count(".") >= 2 and any(len(seg) > 2 for seg in w.split(".")):
            return False  # abbreviation chain, e.g. "Cert.Sci.Nat" (but not "P.J.")
        core = w.strip(",.'’")
        if not core:
            return False
        if core.lower() in TITLE_WORDS:
            return False
        if core.lower() in LOWERCASE_PARTICLES:
            continue
        if require_leading_upper:
            if not core[0].isupper():
                return False
        elif not any(ch.isupper() for ch in core):
            return False
    return True


def _clean(spans: set[str], is_name_pool: bool = False) -> list[str]:
    seen = {}
    for s in spans:
        s = s.strip()
        if len(s) < MIN_TOKEN_LEN or any(len(t) < 1 for t in s.split()):
            continue

        if is_name_pool:
            tokens = _strip_honorifics(s.split())
            tokens = [_strip_prefix(t) for t in tokens]
            s = " ".join(tokens).strip()
            if not s:
                continue

        if not _looks_like_proper_noun(s, require_leading_upper=is_name_pool):
            continue

        seen.setdefault(s.lower(), s)  # case-insensitive dedup, keep first casing seen
    return sorted(seen.values())


def _spans_from_tokens_tags(tokens: list[str], tags: list[str], entity_prefix: str) -> set[str]:
    """Join contiguous B-<prefix>/I-<prefix> token spans into strings."""
    spans = set()
    current: list[str] = []
    for tok, tag in zip(tokens, tags):
        if tag == f"B-{entity_prefix}":
            if current:
                spans.add(" ".join(current))
            current = [tok]
        elif tag == f"I-{entity_prefix}" and current:
            current.append(tok)
        else:
            if current:
                spans.add(" ".join(current))
            current = []
    if current:
        spans.add(" ".join(current))
    return spans


def _detect_prefix(tag_names: list[str], keywords: tuple[str, ...]) -> str | None:
    """Find the B-<X> tag whose <X> contains one of `keywords` (e.g. PER/PERS/PERSON)."""
    for tag in tag_names:
        if not tag.startswith("B-"):
            continue
        suffix = tag[2:]
        if any(kw in suffix.upper() for kw in keywords):
            return suffix
    return None


def extract_hf_dataset(dataset_dir: Path) -> tuple[set[str], set[str]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_dir))
    split = ds["train"] if "train" in ds else next(iter(ds.values()))
    tag_names = split.features["ner_tags"].feature.names

    person_prefix = _detect_prefix(tag_names, ("PER",))
    loc_prefix = _detect_prefix(tag_names, ("LOC",))
    if person_prefix is None or loc_prefix is None:
        raise ValueError(f"could not detect PER/LOC tag prefixes in {tag_names} for {dataset_dir}")

    names, places = set(), set()
    for row in split:
        tags = [tag_names[t] for t in row["ner_tags"]]
        tokens = row["tokens"]
        names |= _spans_from_tokens_tags(tokens, tags, person_prefix)
        places |= _spans_from_tokens_tags(tokens, tags, loc_prefix)
    return names, places


def _read_conll_sentences(path: Path) -> list[tuple[list[str], list[str]]]:
    sentences = []
    tokens, tags = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            if tokens:
                sentences.append((tokens, tags))
            tokens, tags = [], []
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        tokens.append(parts[0])
        tags.append(parts[-1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def extract_conll_files(paths: list[Path]) -> tuple[set[str], set[str]]:
    """Extract PER/LOC spans from one or more CoNLL-format files, auto-detecting the
    exact tag prefix (PER vs PERS) from the tags actually present."""
    sentences = []
    all_tags = set()
    for path in paths:
        for tokens, tags in _read_conll_sentences(path):
            sentences.append((tokens, tags))
            all_tags.update(tags)

    person_prefix = _detect_prefix(sorted(all_tags), ("PER",))
    loc_prefix = _detect_prefix(sorted(all_tags), ("LOC",))
    if person_prefix is None or loc_prefix is None:
        raise ValueError(f"could not detect PER/LOC tag prefixes in {sorted(all_tags)} for {paths}")

    names, places = set(), set()
    for tokens, tags in sentences:
        names |= _spans_from_tokens_tags(tokens, tags, person_prefix)
        places |= _spans_from_tokens_tags(tokens, tags, loc_prefix)
    return names, places


def write_pool(group: str, names: set[str], places: set[str]) -> None:
    DATA_POOLS.mkdir(parents=True, exist_ok=True)
    names_clean = _clean(names, is_name_pool=True)
    places_clean = _clean(places, is_name_pool=False)
    (DATA_POOLS / f"names_{group}.txt").write_text("\n".join(names_clean) + "\n")
    (DATA_POOLS / f"places_{group}.txt").write_text("\n".join(places_clean) + "\n")
    print(f"[{group}] {len(names_clean)} names, {len(places_clean)} places")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    # SADiLaR/NCHLT CoNLL-.txt groups: isiZulu, isiXhosa, Siswati standalone; Sesotho+Setswana merged.
    conll_groups = {
        "isiZulu": ["isiZulu"],
        "isiXhosa": ["isiXhosa"],
        "Sesotho_Setswana": ["Sesotho", "Setswana"],
        "Siswati": ["Siswati"],
    }
    for group, langs in conll_groups.items():
        paths = []
        for lang in langs:
            path = DATA_RAW / "nwu-ctext" / lang / "corpus.txt"
            if not path.exists():
                print(f"[warn] missing {path}, run fetch_sources.py first -- skipping {lang}")
                continue
            paths.append(path)
        if not paths:
            continue
        names, places = extract_conll_files(paths)
        write_pool(group, names, places)

    # Afrikaans <- nwu-ctext parquet (HF datasets, no loading script needed)
    afrikaans_dir = DATA_RAW / "nwu-ctext" / "Afrikaans"
    if afrikaans_dir.exists():
        names, places = extract_hf_dataset(afrikaans_dir)
        write_pool("Afrikaans", names, places)
    else:
        print(f"[warn] missing {afrikaans_dir}, run fetch_sources.py first -- skipping Afrikaans")

    # Tshivenda_Xitsonga <- MphayaNER
    mphaya_dir = DATA_RAW / "mphayaner"
    if mphaya_dir.exists():
        mphaya_files = list(mphaya_dir.rglob("*.txt")) + list(mphaya_dir.rglob("*.conll"))
        names, places = extract_conll_files(mphaya_files)
        write_pool("Tshivenda_Xitsonga", names, places)
    else:
        print(f"[warn] missing {mphaya_dir}, run fetch_sources.py first -- skipping Tshivenda_Xitsonga")

    # Western_control <- WikiANN English
    wikiann_dir = DATA_RAW / "wikiann-en"
    if wikiann_dir.exists():
        names, places = extract_hf_dataset(wikiann_dir)
        write_pool("Western_control", names, places)
    else:
        print(f"[warn] missing {wikiann_dir}, run fetch_sources.py first -- skipping Western_control")


if __name__ == "__main__":
    main()
