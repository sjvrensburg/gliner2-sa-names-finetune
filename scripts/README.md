# scripts

- `export_gliner2_onnx_fragments_v2.py` — vendored verbatim from
  [`SemplificaAI/gliner2-rs`](https://github.com/SemplificaAI/gliner2-rs)'s
  `onnx_conversion_scripts/` (Apache-2.0). This is the **only** exporter known to produce the
  8-fragment ONNX split privacy-ext's `gliner2-rs` inference engine expects
  (`encoder`, `token_gather`, `span_rep`, `schema_gather`, `count_pred_argmax`,
  `count_lstm_fixed`, `scorer`, `classifier`, each in fp32/fp16/fp16_iobinding variants).
  Do not substitute a different GLiNER2 ONNX exporter (e.g. `lmoe/gliner2-onnx`) — that project
  uses an incompatible 4-fragment split (`encoder`/`classifier`/`span_rep`/`count_embed`) and
  its output will not load in privacy-ext's server.

  Usage:
  ```sh
  python export_gliner2_onnx_fragments_v2.py --model_path <checkpoint_or_hf_id> --out_dir <out_dir>
  ```
  Requires: `torch`, `onnx`, `onnxruntime`, `huggingface_hub`, and the `gliner2` package
  (`from gliner2 import Extractor`).
