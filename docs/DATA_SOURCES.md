# Data sources for SA name / NER training data

Researched while scoping this project. Not yet vetted for quality in depth — verify
licenses/coverage yourself before committing to one as the backbone of the training set.

## Liberally licensed, usable now (including commercial use)

- **nwu-ctext NER corpora** on Hugging Face — `isizulu_ner_corpus`, `isixhosa_ner_corpus`,
  `sesotho_ner_corpus`, `setswana_ner_corpus`, `siswati_ner_corpus`, `afrikaans_ner_corpus`.
  **License: CC-BY 2.5 South Africa** (attribution required, commercial use OK). Directly
  downloadable, no gate. Best starting point — covers most of the affected name groups.
- **MphayaNER** (Tshivenda) — [github.com/rendanim/MphayaNER](https://github.com/rendanim/MphayaNER).
  **License: Apache 2.0**, fully permissive. Covers Tshivenda/Xitsonga, the worst-performing
  group in the eval (91.4%).

Together, nwu-ctext + MphayaNER cover isiZulu, isiXhosa, Sesotho, Setswana, Siswati,
Afrikaans, and Tshivenda — commercial-use-safe, no NC restriction.

- **WikiANN / PAN-X**, English config — [huggingface.co/datasets/unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann).
  Wikipedia-derived, IOB2-tagged PER/LOC/ORG spans, same shape as the nwu-ctext corpora above.
  **License: CC BY-SA** (Wikipedia's license) — attribution + share-alike apply to
  *redistributing the corpus itself*; extracting individual name/place strings into a
  synthetic training corpus (as this project does, same as the other sources here) is not a
  redistribution of the corpus. Used as the **Western_control** name/place pool — this is the
  standard, licensed source for that group; don't hand-pick or hand-build a Western name list
  when a real corpus (with the same PER/LOC tagging as the SA corpora) is directly available.

## Usable, but restricted

- **MasakhaNER 2.0** — [huggingface.co/datasets/masakhane/masakhaner2](https://huggingface.co/datasets/masakhane/masakhaner2).
  **License: CC-BY-NC 4.0 (non-commercial)**. Fine for eval/research use; NOT safe to fine-tune
  a shipped/commercial model on without separate clearance from Masakhane.
- **SADiLaR / NCHLT NER corpora** — [repo.sadilar.org](https://repo.sadilar.org). Same
  underlying CTexT data as the nwu-ctext HF mirrors, but gated behind a click-through
  terms-of-use. Only worth going through the gate for languages not already on HF
  (Xitsonga, isiNdebele, Sepedi).

## Not real datasets — informal name lists only

Useful for bulking out a template-fill/synthetic corpus, not for redistribution or treating
as a labeled dataset:

- Behind the Name's South African/Zulu/Xhosa/Tswana submitted-name pages
- FamilySearch's South Africa naming-customs page
- Community-compiled name lists (e.g. crowdsourced Zulu surname documents)

## Recommendation

Start with nwu-ctext (largest, cleanest, permissive) + MphayaNER for Tshivenda coverage, plus
WikiANN English for the Western_control group. Extract PERSON- and LOCATION-tagged spans from
these corpora as your real name/place pools (see `scripts/corpus/extract_pools.py`), then
generate the `address_attn`-style training examples synthetically (see
`docs/TRAINING_DATA_FORMAT.md`) — the corpora's original sentences are useful too, but the
*specific* context/phrasing gap this project targets isn't guaranteed to already exist in
them, so synthetic generation is still needed for the failure mode itself.
