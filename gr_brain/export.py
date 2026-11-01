"""A training run -> ``model.onnx`` the Brain simulator can run, checked before upload.

The exported graph has ONE float32 input ``observation`` of shape [1, inputs]
(already normalised by the platform, per the adapter) and ONE float32 output
``action`` of shape [1, chunk x joints] in LeRobot units - what Brain accepts.
Before anything is written as "ready", onnxruntime runs the file and its output
is compared with PyTorch's on random inputs.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from gr_brain.adapter import input_size, output_size, validate_adapter
from gr_brain.platform import robot as robot_info
from gr_brain.train import _torch, load_checkpoint, rebuild

MAX_MODEL_BYTES = 16 * 1024 * 1024   # the platform's upload limit
PARITY_TOL = 1e-3


@dataclass
class ExportResult:
    onnx_path: Path
    adapter: Dict[str, Any]
    info: Dict[str, Any]
    size_bytes: int
    max_abs_diff: float


def _export(module, dummy, path: Path) -> None:
    torch = _torch()
    kw = dict(input_names=["observation"], output_names=["action"])
    quiet = logging.getLogger("torch.onnx")
    level = quiet.level
    quiet.setLevel(logging.ERROR)          # "torchvision is not installed..." etc.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                torch.onnx.export(module, (dummy,), str(path), opset_version=18, dynamo=True, verbose=False, **kw)
            except Exception:  # noqa: BLE001 - older torch or no onnxscript: the TorchScript exporter
                torch.onnx.export(module, (dummy,), str(path), opset_version=17, dynamo=False, **kw)
    finally:
        quiet.setLevel(level)
    _embed_weights(path)


def _embed_weights(path: Path) -> None:
    """One self-contained file. The newer exporter can write the weights to a side
    file (``model.onnx.data``); Brain uploads ONE file, so fold them back in."""
    side = [p for p in path.parent.glob(path.name + "*") if p != path]
    if not side:
        return
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(model, str(path), save_as_external_data=False)
    for p in side:
        p.unlink(missing_ok=True)


def export_run(run_dir: Union[str, Path]) -> ExportResult:
    torch = _torch()
    import onnxruntime as ort

    run = Path(run_dir)
    ck = load_checkpoint(run)
    adapter = ck["adapter"]
    problems = validate_adapter(adapter, robot_info(ck["robot"]))
    if problems:
        raise ValueError("the run's adapter is not valid: " + " ".join(problems))
    module = rebuild(ck)
    n_in, n_out = input_size(adapter), output_size(adapter)
    path = run / "model.onnx"
    _export(module, torch.zeros(1, n_in), path)

    # From the BYTES, exactly as Brain will receive it: a model that still leans on
    # a side file next to it fails here, not after upload.
    sess = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    inputs, outputs = sess.get_inputs(), sess.get_outputs()
    if len(inputs) != 1 or not outputs:
        raise RuntimeError(f"export produced {len(inputs)} inputs / {len(outputs)} outputs; Brain needs exactly one input")
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((64, n_in)).astype(np.float32)
    diff = 0.0
    with torch.no_grad():
        for row in probe:
            x = row.reshape(1, -1)
            got = sess.run([outputs[0].name], {inputs[0].name: x})[0]
            if got.size != n_out:
                raise RuntimeError(f"the model returns {got.size} values; the adapter expects {n_out}")
            want = module(torch.as_tensor(x)).numpy()
            diff = max(diff, float(np.abs(got - want).max()))
    if not np.isfinite(diff) or diff > PARITY_TOL:
        raise RuntimeError(f"onnxruntime and PyTorch disagree by {diff:.2e} (tolerance {PARITY_TOL:.0e})")
    size = path.stat().st_size
    if size > MAX_MODEL_BYTES:
        raise RuntimeError(f"model.onnx is {size / 1e6:.1f} MB; Brain accepts up to 16 MB - train a smaller network")

    def shape(v):
        return [d if isinstance(d, int) else str(d) for d in v.shape]
    info = {"inputName": inputs[0].name, "outputName": outputs[0].name,
            "inputShape": shape(inputs[0]), "outputShape": shape(outputs[0])}
    (run / "adapter.json").write_text(json.dumps(adapter, indent=2), encoding="utf-8")
    (run / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return ExportResult(path, adapter, info, size, diff)
