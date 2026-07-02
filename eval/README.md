# eval

Eval-set generator and runner used to measure the address-line detection gap described in the
top-level README (carried over from privacy-ext's `server/eval/`, where the original run
produced the 266-case / 95.9%-overall / 73.7%-address_attn numbers quoted there). The generated
dataset itself (`sa_names_eval.jsonl`) is not checked in — regenerate it locally.

- `sa_names_eval_gen.py` — generates `sa_names_eval.jsonl` from a name pool × context-template
  matrix. If `data/pools/eval/` exists (built by `scripts/corpus/split_eval_holdout.py`), it
  samples up to `NAMES_PER_GROUP` (40) real names per group from there instead of the original
  ~40-name hardcoded list, falling back to that list with a warning if the pools aren't built
  yet. `data/pools/eval/` is a **held-out slice**, disjoint from every training pool
  (`data/pools/train/`) including cross-group leaks (e.g. a surname shared between isiZulu and
  Siswati pools is excluded from all training pools once it lands in any eval-holdout slice) —
  so eval recall isn't measuring memorization of names the model was fine-tuned on. Regenerate
  the eval pools from scratch with `fetch_sources.py` → `extract_pools.py` →
  `split_eval_holdout.py` before regenerating training data, so the split stays in sync.
- `run_sa_names_eval.py` — POSTs each case to a running privacy-ext `pii-server`
  (`http://127.0.0.1:8731/classify`) and reports recall by name group, by context, and the
  worst (group, context) cells. Usage: `python sa_names_eval_gen.py && python run_sa_names_eval.py <bearer_token>`.

Run this against both the currently-shipped fragments and any newly fine-tuned/exported
fragments to confirm the fine-tune actually improved address-line recall without regressing
anything else.
