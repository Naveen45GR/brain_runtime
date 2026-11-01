import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from gr_brain.adapter import validate_adapter  # noqa: E402
from gr_brain.dataset import load, to_arrays  # noqa: E402
from gr_brain.export import export_run  # noqa: E402
from gr_brain.platform import robot  # noqa: E402
from gr_brain.train import TrainConfig, train  # noqa: E402


@pytest.mark.parametrize("chunk", [1, 4])
def test_train_then_export_runs_in_onnxruntime(tmp_path, dataset_doc, chunk):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(dataset_doc))
    res = train(path, tmp_path / "run", TrainConfig(epochs=120, hidden=64, chunk=chunk, device="cpu"), log=lambda _m: None)
    assert res.val_mae < 15.0, res          # the synthetic task is learnable (degrees)
    adapter = json.loads((tmp_path / "run" / "adapter.json").read_text())
    assert validate_adapter(adapter, robot("koch-v1-1")) == []
    assert adapter["action"]["chunk"] == chunk and adapter["hz"] == 25 and "normalize" in adapter

    ex = export_run(tmp_path / "run")
    assert ex.max_abs_diff < 1e-3 and ex.size_bytes < 16 * 1024 * 1024
    # ONE self-contained file - Brain uploads only model.onnx (no model.onnx.data side file).
    assert [p.name for p in (tmp_path / "run").glob("model.onnx*")] == ["model.onnx"]
    ort.InferenceSession(ex.onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
    assert ex.info["inputName"] == "observation" and ex.info["outputName"] == "action"

    # Feed it exactly as the Brain simulator does: normalise with the adapter, then run.
    sess = ort.InferenceSession(str(ex.onnx_path), providers=["CPUExecutionProvider"])
    ds = load(dataset_doc)
    arr = to_arrays(ds, hz=25, chunk=chunk)
    mean, std = np.asarray(adapter["normalize"]["mean"]), np.asarray(adapter["normalize"]["std"])
    x = ((arr.X[10] - mean) / std).astype(np.float32).reshape(1, -1)
    out = sess.run(None, {"observation": x})[0].reshape(-1)
    J = len(adapter["action"]["joints"])
    assert out.shape == (chunk * J,) and np.all(np.isfinite(out))
    assert np.abs(out[:J] - arr.Y[10, :J]).mean() < 20.0


def test_export_refuses_a_folder_that_is_not_a_run(tmp_path):
    (tmp_path / "checkpoint.pt").write_bytes(b"")
    with pytest.raises(Exception):
        export_run(tmp_path)
