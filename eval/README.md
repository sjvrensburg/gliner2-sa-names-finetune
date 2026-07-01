# eval

Eval-set generator and runner used to measure the address-line detection gap described in the
top-level README (carried over from privacy-ext's `server/eval/`, where the original run
produced the 266-case / 95.9%-overall / 73.7%-address_attn numbers quoted there). The generated
dataset itself (`sa_names_eval.jsonl`) is not checked in — regenerate it locally.

- `sa_names_eval_gen.py` — generates `sa_names_eval.jsonl` from a name pool × context-template
  matrix. Run it as-is to reproduce the baseline eval set, or **extend it with a larger, real
  name list** (see `../docs/DATA_SOURCES.md`) before using it as the fine-tuning regression
  gate — the built-in ~40-name list was a quick first pass, not meant to be the final eval set.
- `run_sa_names_eval.py` — POSTs each case to a running privacy-ext `pii-server`
  (`http://127.0.0.1:8731/classify`) and reports recall by name group, by context, and the
  worst (group, context) cells. Usage: `python sa_names_eval_gen.py && python run_sa_names_eval.py <bearer_token>`.

Run this against both the currently-shipped fragments and any newly fine-tuned/exported
fragments to confirm the fine-tune actually improved address-line recall without regressing
anything else.
