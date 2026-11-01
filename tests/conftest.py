import math
from typing import Any, Dict, List

import pytest

from gr_brain.platform import robot

ROBOT = "koch-v1-1"


def synthetic_dataset(episodes: int = 4, seconds: float = 4.0, fps: int = 30, robot_id: str = ROBOT) -> Dict[str, Any]:
    """Smooth demonstrations where the commanded action leads the observation -
    a mapping a small network can learn."""
    joints = robot(robot_id).model_joints
    eps: List[Dict[str, Any]] = []
    for e in range(episodes):
        frames = []
        for i in range(int(seconds * fps) + 1):
            t = i / fps
            obs, act = {}, {}
            for k, j in enumerate(joints):
                phase = 0.7 * e + 0.9 * k
                amp = 40.0 if j.kind == "revolute" else 30.0
                centre = 50.0 if j.kind == "gripper" else 0.0
                obs[j.key] = round(centre + amp * math.sin(1.3 * t + phase), 4)
                act[j.key] = round(centre + amp * math.sin(1.3 * (t + 0.1) + phase), 4)
            frames.append({"t": round(t, 4), "observation": obs, "action": act})
        eps.append({"id": f"ep{e}", "index": e, "task": "wave", "fps": fps, "frames": frames})
    return {"format": "gr-brain-dataset/1", "dataset": "demo", "robot": robot_id, "fps": fps,
            "episodes": eps, "features": {}}


@pytest.fixture
def dataset_doc():
    return synthetic_dataset()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GR_BRAIN_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("GR_BRAIN_TOKEN", raising=False)
    monkeypatch.delenv("GR_BRAIN_API", raising=False)
