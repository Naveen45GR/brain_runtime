"""What the platform defines: robots, adapter rules, how a model run is judged.

Read from ``gr_brain/data/platform.json``, which the platform generates from its
own robot and task definitions - so a joint name or a criterion here is exactly
the one the Brain simulator and the server verifier use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Joint:
    name: str          #: the simulator's joint name - what a model adapter refers to
    lerobot: str       #: LeRobot's name - what recordings use (``<lerobot>.pos``)
    kind: str          #: revolute | gripper | prismatic | wheel
    reversed: bool = False

    @property
    def key(self) -> str:
        """The recording key for this joint's position."""
        return f"{self.lerobot}.pos"


@dataclass(frozen=True)
class Robot:
    id: str
    joints: tuple
    base: bool
    tasks: tuple

    @property
    def model_joints(self) -> List[Joint]:
        """Joints a model may see or drive (wheels excluded, as on the platform)."""
        return [j for j in self.joints if j.kind != "wheel"]

    def by_key(self) -> Dict[str, Joint]:
        return {j.key: j for j in self.model_joints}

    def task(self, task_id: str) -> Dict[str, Any]:
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        raise KeyError(f"{self.id} has no task {task_id!r} (has: {', '.join(t['id'] for t in self.tasks) or 'none'})")


@lru_cache(maxsize=1)
def platform() -> Dict[str, Any]:
    raw = resources.files("gr_brain").joinpath("data/platform.json").read_text(encoding="utf-8")
    doc = json.loads(raw)
    if doc.get("format") != "gr-brain-platform/1":
        raise RuntimeError(f"unsupported platform data format {doc.get('format')!r}")
    return doc


ROBOT_ALIASES = {"hello-stretch-4": "hello-robot-stretch-4"}


def robot_ids() -> List[str]:
    return sorted(platform()["robots"])


def robot(robot_id: str) -> Robot:
    rid = ROBOT_ALIASES.get(robot_id, robot_id)
    spec = platform()["robots"].get(rid)
    if spec is None:
        raise KeyError(f"unknown robot {robot_id!r}; known: {', '.join(robot_ids())}")
    joints = tuple(Joint(j["name"], j["lerobot"], j["kind"], bool(j.get("reversed"))) for j in spec["joints"])
    return Robot(rid, joints, bool(spec.get("base")), tuple(spec.get("tasks") or ()))


def adapter_rules() -> Dict[str, Any]:
    return platform()["adapter"]


_OPS = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def failed_criteria(criteria: List[Dict[str, Any]], measures: Dict[str, float]) -> List[str]:
    """Which pass criteria a run's measures miss, as readable lines."""
    out = []
    for c in criteria:
        value = measures.get(c["measure"])
        op = _OPS.get(c["op"])
        if value is None or op is None:
            continue
        if not op(value, c["value"]):
            out.append(f"{c['measure']} = {value:g} (needs {c['op']} {c['value']:g}) - {c.get('why', '')}".rstrip(" -"))
    return out


def model_spec(model_id: str, robot_id: str, max_sim_time: Optional[float] = None, seed: int = 1,
               task_id: Optional[str] = None) -> Dict[str, Any]:
    """The experiment spec a model run is judged by - the same as the Brain page builds."""
    doc = platform()
    r = robot(robot_id)
    task = r.task(task_id) if task_id else None
    criteria = list(doc["model_criteria"])
    spec: Dict[str, Any] = {
        "id": f"model:{model_id}",
        "name": "Model run",
        "robot": r.id,
        "code": "",
        "seed": seed,
        "maxSimTime": float(max_sim_time if max_sim_time is not None else doc["model_run_seconds"]),
        "mode": "monitor",
        "pass": ([doc["task_criterion"]] + criteria) if task else criteria,
    }
    if task:
        spec["task"] = task
    return spec
