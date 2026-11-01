"""A custom policy for `gr-brain serve` - anything that maps observation -> action.

    gr-brain serve examples/custom_policy.py

Replace `act()` with your model: a LeRobot policy, a diffusion policy, a VLA on your
GPU... The adapter tells Brain what to send and how to read what you return
(docs/adapter.md). This example gently sweeps the Koch arm's shoulder joint while
holding the others where they are - a safe first run.
"""

import math

import numpy as np

from gr_brain.adapter import make_adapter
from gr_brain.platform import robot

ROBOT = "koch-v1-1"
JOINTS = [j.name for j in robot(ROBOT).model_joints]
HZ = 25


class SweepShoulder:
    robot = ROBOT
    name = "sweep-shoulder"

    def __init__(self, device=None):
        self.device = device or "cpu"
        # Positions of every joint in (LeRobot units: degrees / gripper %), absolute targets out.
        self.adapter = make_adapter(JOINTS, hz=HZ, chunk=1, max_deg_per_sec=60)
        self.t = 0.0
        self.start = None

    def reset(self, seed: int) -> None:
        self.t = 0.0
        self.start = None

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.start is None:
            self.start = observation.copy()       # hold everything where it started...
        target = self.start.copy()
        target[1] = self.start[1] + 20.0 * math.sin(2 * math.pi * 0.2 * self.t)   # ...and sweep joint 1 +/-20 deg
        self.t += 1.0 / HZ
        return target.astype(np.float32)


def make_policy(device=None):
    return SweepShoulder(device)
