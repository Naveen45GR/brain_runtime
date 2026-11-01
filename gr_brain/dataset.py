"""Brain recordings (format ``gr-brain-dataset/1``) -> training arrays.

A recording frame is ``{t, observation: {<joint>.pos: v}, action: {<joint>.pos: v}}``
in LeRobot keys and units (revolute: degrees, gripper: 0-100 %, prismatic: metres).
``observation`` is what the joints did, ``action`` what the person commanded.

For training, each episode is resampled onto the model's control rate (the
Brain simulator calls a model at 5/10/25/50 Hz) by holding the latest frame,
and each sample's target is the next ``chunk`` commanded actions (the last one
repeated past the end of the episode).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from gr_brain.platform import Joint, Robot, adapter_rules, robot as robot_info

FORMAT = "gr-brain-dataset/1"


@dataclass
class Episode:
    id: str
    t: np.ndarray                      # [F] seconds
    observation: List[Dict[str, float]]
    action: List[Dict[str, float]]
    task: str = ""


@dataclass
class Dataset:
    name: str
    robot: Robot
    fps: float
    episodes: List[Episode] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return sum(len(e.t) for e in self.episodes)


def load(source: Union[str, Path, Dict[str, Any]]) -> Dataset:
    """Read a downloaded dataset (a path to the JSON, or the parsed document)."""
    doc = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"not a Brain dataset (format {doc.get('format')!r}, expected {FORMAT!r})")
    r = robot_info(doc["robot"])
    episodes = []
    for i, ep in enumerate(doc.get("episodes") or []):
        frames = ep.get("frames") or []
        if len(frames) < 2:
            continue
        episodes.append(Episode(
            id=str(ep.get("id") or i),
            t=np.asarray([float(f["t"]) for f in frames], dtype=np.float64),
            observation=[f.get("observation") or {} for f in frames],
            action=[f.get("action") or {} for f in frames],
            task=str(ep.get("task") or ""),
        ))
    if not episodes:
        raise ValueError("the dataset has no episodes with at least two frames")
    return Dataset(name=str(doc.get("dataset") or "dataset"), robot=r, fps=float(doc.get("fps") or 30), episodes=episodes)


def usable_joints(ds: Dataset) -> List[Joint]:
    """Model joints present in EVERY frame's observation and action, in robot order."""
    out = []
    for j in ds.robot.model_joints:
        if all(j.key in o and j.key in a for e in ds.episodes for o, a in zip(e.observation, e.action)):
            out.append(j)
    if not out:
        raise ValueError("no joint is recorded in every frame - record with the robot's arm joints")
    return out


def default_hz(fps: float) -> int:
    """The fastest control rate the recording can support (a model can't be fed faster than it was taught)."""
    allowed = sorted(adapter_rules()["hz"])
    ok = [h for h in allowed if h <= fps + 1e-6]
    return min(25, ok[-1]) if ok else allowed[0]


@dataclass
class Arrays:
    joints: List[Joint]
    hz: int
    chunk: int
    X: np.ndarray            # [N, J] observation (LeRobot units)
    Y: np.ndarray            # [N, chunk*J] next `chunk` commanded actions
    episode: np.ndarray      # [N] episode index of each sample


def to_arrays(ds: Dataset, *, hz: Optional[int] = None, chunk: int = 1,
              joints: Optional[Sequence[Joint]] = None) -> Arrays:
    if hz is None:
        hz = default_hz(ds.fps)
    if hz not in adapter_rules()["hz"]:
        raise ValueError(f"hz must be one of {adapter_rules()['hz']}")
    if not 1 <= chunk <= adapter_rules()["max_chunk"]:
        raise ValueError(f"chunk must be 1-{adapter_rules()['max_chunk']}")
    joints = list(joints) if joints else usable_joints(ds)
    keys = [j.key for j in joints]
    xs, ys, eps = [], [], []
    for ei, e in enumerate(ds.episodes):
        obs = np.asarray([[o[k] for k in keys] for o in e.observation], dtype=np.float32)
        act = np.asarray([[a[k] for k in keys] for a in e.action], dtype=np.float32)
        t = e.t - e.t[0]
        steps = np.arange(0.0, t[-1] + 1e-9, 1.0 / hz)
        # Hold the latest frame at or before each control step.
        idx = np.clip(np.searchsorted(t, steps + 1e-9, side="right") - 1, 0, len(t) - 1)
        o, a = obs[idx], act[idx]
        n = len(steps)
        future = np.minimum(np.arange(n)[:, None] + np.arange(chunk)[None, :], n - 1)   # [n, chunk]
        xs.append(o)
        ys.append(a[future].reshape(n, chunk * len(keys)))
        eps.append(np.full(n, ei, dtype=np.int32))
    return Arrays(joints, hz, chunk, np.concatenate(xs), np.concatenate(ys), np.concatenate(eps))


def save_npz(arrays: Arrays, path: Union[str, Path]) -> Path:
    path = Path(path)
    np.savez_compressed(path, X=arrays.X, Y=arrays.Y, episode=arrays.episode,
                        joints=np.asarray([j.name for j in arrays.joints]),
                        keys=np.asarray([j.key for j in arrays.joints]), hz=arrays.hz, chunk=arrays.chunk)
    return path
