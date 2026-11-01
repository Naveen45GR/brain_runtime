# Changelog

## 0.1.0 — 2026-10

First release.

* `gr-brain login / projects / datasets / pull` — your Brain recordings (gr-brain-dataset/1).
* `gr-brain train` — behaviour cloning (MLP, action chunking) on CUDA, Apple MPS or CPU.
* `gr-brain export` — one self-contained ONNX file, checked against PyTorch with onnxruntime.
* `gr-brain upload [--verify]` and `gr-brain verify` — add a model to a project and have the
  platform re-run it on its own servers (`model verified`).
* `gr-brain serve` — run an ONNX model, a training run or your own Python policy live against
  the Brain simulator through the platform relay (protocol v1).
* `gr-brain doctor`.
