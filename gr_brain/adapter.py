"""The model adapter: how a model's tensors map onto a robot's joints.

Stored with every uploaded model; the Brain simulator and the server verifier
apply it around each inference (docs/adapter.md). In short:

* input - the listed joints' position (or velocity), in LeRobot units, then
  ``(x - mean) / std`` when ``normalize`` is given;
* output - ``chunk x len(action.joints)`` values, times ``scale``: absolute
  targets (or deltas) in LeRobot units, one chunk step per control period;
* safety (applied by the platform, not the model) - every target is clamped to
  the joint's range and rate-limited to ``safety.maxDegPerSec``; a non-finite
  output stops the run before anything moves.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from gr_brain.platform import Robot, adapter_rules


def make_adapter(joints: Sequence[str], *, hz: int, chunk: int = 1, mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None, mode: str = "absolute", scale: float = 1.0,
                 max_deg_per_sec: float = 180.0, observe: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """An adapter for a state-based policy: positions of ``observe`` (default: ``joints``) in,
    targets for ``joints`` out."""
    observe = list(observe) if observe is not None else list(joints)
    adapter: Dict[str, Any] = {
        "version": 1,
        "hz": int(hz),
        "observation": [{"joint": j, "field": "pos"} for j in observe],
        "units": "lerobot",
        "action": {"joints": list(joints), "mode": mode, "chunk": int(chunk), "scale": float(scale)},
        "safety": {"maxDegPerSec": float(max_deg_per_sec)},
    }
    if mean is not None or std is not None:
        adapter["normalize"] = {"mean": [float(x) for x in mean or []], "std": [float(x) for x in std or []]}
    return adapter


def validate_adapter(a: Dict[str, Any], robot: Robot) -> List[str]:
    """Problems as sentences (empty when usable) - the same rules the Brain page applies."""
    rules = adapter_rules()
    known = {j.name for j in robot.model_joints}
    errors: List[str] = []
    if a.get("version") != 1:
        errors.append("Unknown adapter version.")
    if a.get("hz") not in rules["hz"]:
        errors.append(f"Rate must be one of {', '.join(map(str, rules['hz']))} Hz.")
    obs = a.get("observation") or []
    if not obs:
        errors.append("The model needs at least one input.")
    for o in obs:
        if o.get("joint") not in known:
            errors.append(f'Input joint "{o.get("joint")}" is not on {robot.id}.')
        if o.get("field") not in ("pos", "vel"):
            errors.append(f'Input field must be "pos" or "vel", not {o.get("field")!r}.')
    if a.get("units") not in ("lerobot", "sim"):
        errors.append('Units must be "lerobot" or "sim".')
    act = a.get("action") or {}
    joints = act.get("joints") or []
    if not joints:
        errors.append("The model must drive at least one joint.")
    if len(set(joints)) != len(joints):
        errors.append("An output joint is listed twice.")
    for j in joints:
        if j not in known:
            errors.append(f'Output joint "{j}" is not on {robot.id}.')
    if act.get("mode") not in ("absolute", "delta"):
        errors.append('Action mode must be "absolute" or "delta".')
    chunk = act.get("chunk")
    if not isinstance(chunk, int) or not 1 <= chunk <= rules["max_chunk"]:
        errors.append(f"Chunk must be 1-{rules['max_chunk']}.")
    scale = act.get("scale")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale == 0:
        errors.append("Output scale must be a non-zero number.")
    lo, hi = rules["safety_max_deg_per_sec"]
    speed = (a.get("safety") or {}).get("maxDegPerSec")
    if not isinstance(speed, (int, float)) or not lo <= speed <= hi:
        errors.append(f"Speed limit must be {lo}-{hi} deg/s.")
    norm = a.get("normalize")
    if norm:
        n = len(obs)
        mean, std = norm.get("mean") or [], norm.get("std") or []
        if len(mean) != n or len(std) != n:
            errors.append(f"Normalisation needs {n} means and {n} standard deviations (one per input).")
        elif any(not math.isfinite(m) for m in mean) or any(not math.isfinite(s) or abs(s) <= 1e-12 for s in std):
            errors.append("Normalisation values must be finite, and no standard deviation may be zero.")
    return errors


def input_size(a: Dict[str, Any]) -> int:
    return len(a["observation"])


def output_size(a: Dict[str, Any]) -> int:
    return int(a["action"]["chunk"]) * len(a["action"]["joints"])
