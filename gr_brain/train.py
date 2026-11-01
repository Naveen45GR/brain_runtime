"""Behaviour cloning: learn "observation -> next commanded action(s)" from your recordings.

A small MLP, trained with PyTorch on whatever you have (CUDA, Apple MPS, or CPU -
recordings are small; a CPU finishes in minutes). Inputs are normalised with the
dataset's statistics, which go into the model's adapter so the Brain simulator
normalises exactly the same way; outputs are de-normalised inside the network,
so the exported model emits plain LeRobot units.

A run folder (``runs/<name>/``) holds ``checkpoint.pt`` (weights + everything
needed to rebuild and export), ``adapter.json`` and ``metrics.json``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np

from gr_brain.adapter import make_adapter
from gr_brain.dataset import Arrays, Dataset, load, to_arrays


def _torch():
    try:
        import torch
    except ImportError as exc:   # pragma: no cover - depends on the install
        raise RuntimeError("training needs PyTorch: pip install 'gr-brain[train]' "
                           "(pick the build for your GPU at https://pytorch.org first)") from exc
    return torch


@dataclass
class TrainConfig:
    hz: Optional[int] = None          #: control rate the model runs at (default: from the recording)
    chunk: int = 1                    #: actions predicted per inference
    hidden: int = 256
    layers: int = 2
    epochs: int = 400
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.15
    patience: int = 40                #: epochs without validation improvement before stopping
    seed: int = 0
    max_deg_per_sec: float = 180.0    #: the adapter's safety rate limit
    device: Optional[str] = None      #: cuda | mps | cpu (default: best available)


@dataclass
class TrainResult:
    run_dir: Path
    device: str
    epochs: int
    train_loss: float
    val_loss: float
    val_mae: float                    #: mean abs error of the first predicted action, LeRobot units
    samples: int
    seconds: float


def build_mlp(n_in: int, n_out: int, hidden: int, layers: int):
    torch = _torch()
    nn = torch.nn
    dims = [n_in] + [hidden] * layers
    mods = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.ReLU()]
    mods.append(nn.Linear(dims[-1], n_out))
    return nn.Sequential(*mods)


def policy_module(mlp, y_mean: np.ndarray, y_std: np.ndarray):
    """The exported network: normalised observation in, LeRobot-unit actions out."""
    torch = _torch()

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp
            self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32).reshape(1, -1))
            self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32).reshape(1, -1))

        def forward(self, observation):
            return self.mlp(observation) * self.y_std + self.y_mean

    return Policy()


def pick_device(requested: Optional[str] = None) -> str:
    torch = _torch()
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split(arrays: Arrays, fraction: float, rng: np.random.Generator):
    """Hold out whole episodes when there are several (no leakage between neighbours)."""
    n_ep = int(arrays.episode.max()) + 1
    if n_ep >= 3:
        k = max(1, int(round(n_ep * fraction)))
        val_eps = set(rng.choice(n_ep, size=k, replace=False).tolist())
        val = np.isin(arrays.episode, list(val_eps))
    else:
        n = len(arrays.X)
        cut = int(n * (1 - fraction))
        val = np.zeros(n, dtype=bool)
        val[cut:] = True
    if val.all() or not val.any():
        val = np.zeros(len(arrays.X), dtype=bool)
        val[-max(1, len(val) // 7):] = True
    return ~val, val


def train(dataset: Union[str, Path, Dataset], out_dir: Union[str, Path], cfg: Optional[TrainConfig] = None,
          log: Callable[[str], None] = print) -> TrainResult:
    torch = _torch()
    cfg = cfg or TrainConfig()
    ds = dataset if isinstance(dataset, Dataset) else load(dataset)
    arrays = to_arrays(ds, hz=cfg.hz, chunk=cfg.chunk)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)

    tr, va = _split(arrays, cfg.val_fraction, rng)
    x_mean = arrays.X[tr].mean(0)
    x_std = np.maximum(arrays.X[tr].std(0), 1e-3)
    y_mean = arrays.Y[tr].mean(0)
    y_std = np.maximum(arrays.Y[tr].std(0), 1e-3)

    def tensors(mask):
        x = (arrays.X[mask] - x_mean) / x_std
        y = (arrays.Y[mask] - y_mean) / y_std
        return torch.as_tensor(x, dtype=torch.float32, device=device), torch.as_tensor(y, dtype=torch.float32, device=device)

    xt, yt = tensors(tr)
    xv, yv = tensors(va)
    mlp = build_mlp(arrays.X.shape[1], arrays.Y.shape[1], cfg.hidden, cfg.layers).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()

    J = len(arrays.joints)
    log(f"training on {device}: {len(xt)} train / {len(xv)} val samples, {J} joints, {arrays.hz} Hz, chunk {arrays.chunk}")
    started = time.time()
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    train_loss = float("nan")
    epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        mlp.train()
        order = torch.randperm(len(xt), device=device)
        total = 0.0
        for i in range(0, len(xt), cfg.batch):
            idx = order[i:i + cfg.batch]
            opt.zero_grad()
            loss = loss_fn(mlp(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        train_loss = total / max(1, len(xt))
        mlp.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(mlp(xv), yv))
        if val_loss < best - 1e-6:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            log(f"  epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}")
        if stale >= cfg.patience:
            log(f"  stopping early at epoch {epoch} (best val at {best_epoch})")
            break
    if best_state is not None:
        mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        pred = mlp(xv).cpu().numpy() * y_std + y_mean
    true = arrays.Y[va]
    val_mae = float(np.abs(pred[:, :J] - true[:, :J]).mean())

    run = Path(out_dir)
    run.mkdir(parents=True, exist_ok=True)
    joints = [j.name for j in arrays.joints]
    adapter = make_adapter(joints, hz=arrays.hz, chunk=arrays.chunk, mean=x_mean.tolist(), std=x_std.tolist(),
                           max_deg_per_sec=cfg.max_deg_per_sec)
    torch.save({
        "format": "gr-brain-checkpoint/1",
        "robot": ds.robot.id,
        "dataset": ds.name,
        "joints": joints,
        "keys": [j.key for j in arrays.joints],
        "config": asdict(cfg),
        "n_in": int(arrays.X.shape[1]), "n_out": int(arrays.Y.shape[1]),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        "adapter": adapter,
        "state_dict": {k: v.cpu() for k, v in mlp.state_dict().items()},
    }, run / "checkpoint.pt")
    (run / "adapter.json").write_text(json.dumps(adapter, indent=2), encoding="utf-8")
    result = TrainResult(run, device, epoch, train_loss, best, val_mae, int(len(arrays.X)), round(time.time() - started, 1))
    (run / "metrics.json").write_text(json.dumps({**asdict(result), "run_dir": str(run)}, indent=2), encoding="utf-8")
    log(f"done in {result.seconds}s - validation error {val_mae:.3f} (deg / % / m) -> {run}")
    return result


def load_checkpoint(run_dir: Union[str, Path]) -> Dict:
    torch = _torch()
    ck = torch.load(Path(run_dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
    if ck.get("format") != "gr-brain-checkpoint/1":
        raise ValueError(f"{run_dir} is not a gr-brain training run")
    return ck


def rebuild(ck: Dict):
    """The trained network (eval mode, CPU), observation -> LeRobot-unit actions."""
    cfg = ck["config"]
    mlp = build_mlp(ck["n_in"], ck["n_out"], cfg["hidden"], cfg["layers"])
    mlp.load_state_dict(ck["state_dict"])
    return policy_module(mlp, np.asarray(ck["y_mean"]), np.asarray(ck["y_std"])).eval()
