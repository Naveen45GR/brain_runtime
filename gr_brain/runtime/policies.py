"""Load any policy behind one interface: ``act(observation) -> action``.

Three kinds:

* ``model.onnx`` - run with onnxruntime (CUDA if available). Needs its adapter:
  ``adapter.json`` next to it (``gr-brain export`` writes one) or ``--adapter``,
  and the robot (``--robot``, or the training run's checkpoint).
* a training run folder (``runs/<name>/``) - its ``model.onnx`` if exported,
  otherwise the PyTorch checkpoint on your GPU.
* ``my_policy.py`` - your own code: define ``make_policy()`` returning an object
  with ``robot`` (str), ``adapter`` (dict, docs/adapter.md) and
  ``act(observation: np.ndarray) -> np.ndarray``; optionally ``reset(seed)`` and
  ``name``. Anything goes inside - a LeRobot policy, a VLA, a hand-written
  controller - as long as it maps the adapter's inputs to its outputs.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from gr_brain.adapter import input_size, output_size, validate_adapter
from gr_brain.platform import robot as robot_info


@dataclass
class LivePolicy:
    name: str
    kind: str                      # onnx | torch | python
    robot: str
    adapter: Dict[str, Any]
    device: str
    act: Callable[[np.ndarray], np.ndarray]
    reset: Callable[[int], None] = field(default=lambda _seed: None)

    @property
    def inputs(self) -> int:
        return input_size(self.adapter)

    @property
    def outputs(self) -> int:
        return output_size(self.adapter)


def _check(p: LivePolicy) -> LivePolicy:
    problems = validate_adapter(p.adapter, robot_info(p.robot))
    if problems:
        raise ValueError("the policy's adapter is not valid: " + " ".join(problems))
    return p


def _onnx(path: Path, adapter: Dict[str, Any], robot: str, device: Optional[str],
          name: Optional[str] = None) -> LivePolicy:
    import onnxruntime as ort

    available = ort.get_available_providers()
    want_cuda = device in (None, "cuda") and "CUDAExecutionProvider" in available
    providers = (["CUDAExecutionProvider"] if want_cuda else []) + ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    used = sess.get_providers()[0]
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    rank = len(inp.shape)

    def act(obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32).reshape((-1,) if rank == 1 else (1, -1))
        return np.asarray(sess.run([out.name], {inp.name: x})[0], dtype=np.float32).reshape(-1)

    dev = "cuda" if used == "CUDAExecutionProvider" else "cpu"
    return _check(LivePolicy(name or path.stem, "onnx", robot, adapter, dev, act))


def _torch_run(run: Path, device: Optional[str]) -> LivePolicy:
    import torch

    from gr_brain.train import load_checkpoint, pick_device, rebuild

    ck = load_checkpoint(run)
    dev = pick_device(device)
    module = rebuild(ck).to(dev)

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=dev).reshape(1, -1)
            return module(x).cpu().numpy().reshape(-1)

    return _check(LivePolicy(run.name, "torch", ck["robot"], ck["adapter"], dev, act))


def _python(path: Path, device: Optional[str]) -> LivePolicy:
    spec = importlib.util.spec_from_file_location(f"gr_brain_user_policy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    make = getattr(module, "make_policy", None)
    if make is None:
        raise ValueError(f"{path} must define make_policy() - see docs/quickstart.md")
    obj = make(device=device) if "device" in inspect.signature(make).parameters else make()
    for attr in ("robot", "adapter", "act"):
        if not hasattr(obj, attr):
            raise ValueError(f"the policy from {path} has no `{attr}`")
    reset = getattr(obj, "reset", None) or (lambda _seed: None)
    return _check(LivePolicy(str(getattr(obj, "name", path.stem)), "python", obj.robot, dict(obj.adapter),
                             str(getattr(obj, "device", device or "cpu")),
                             lambda o: np.asarray(obj.act(o), dtype=np.float32).reshape(-1), reset))


def load_policy(source: str, *, robot: Optional[str] = None, adapter: Optional[str] = None,
                device: Optional[str] = None) -> LivePolicy:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"{source} not found")
    if path.suffix == ".py":
        return _python(path, device)
    if path.is_dir():
        if (path / "model.onnx").exists():
            from gr_brain.train import load_checkpoint
            ck = load_checkpoint(path)
            ad = json.loads((path / "adapter.json").read_text(encoding="utf-8"))
            return _onnx(path / "model.onnx", ad, robot or ck["robot"], device, name=path.name)
        return _torch_run(path, device)
    if path.suffix == ".onnx":
        ad_path = Path(adapter) if adapter else path.with_name("adapter.json")
        if not ad_path.exists():
            raise ValueError(f"{path.name} needs its adapter: put adapter.json next to it or pass --adapter")
        if not robot:
            ck_path = path.with_name("checkpoint.pt")
            if not ck_path.exists():
                raise ValueError("pass --robot for a model that did not come from `gr-brain train`")
            from gr_brain.train import load_checkpoint
            robot = load_checkpoint(path.parent)["robot"]
        name = path.parent.name if path.stem == "model" and (path.parent / "checkpoint.pt").exists() else None
        return _onnx(path, json.loads(ad_path.read_text(encoding="utf-8")), robot, device, name=name)
    raise ValueError(f"{source}: expected model.onnx, a training run folder, or a .py file")
