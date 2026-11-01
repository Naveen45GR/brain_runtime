# The model adapter

Every model in Brain carries an **adapter**: a small JSON description of how its tensors
map onto the robot's joints. The platform applies it around each inference — the same in
the browser, on the verification servers and with a live runtime.

```json
{
  "version": 1,
  "hz": 25,
  "observation": [
    {"joint": "base_rotation", "field": "pos"},
    {"joint": "pitch", "field": "pos"}
  ],
  "units": "lerobot",
  "normalize": {"mean": [0.0, -12.5], "std": [30.1, 18.4]},
  "action": {"joints": ["base_rotation", "pitch"], "mode": "absolute", "chunk": 1, "scale": 1.0},
  "safety": {"maxDegPerSec": 180}
}
```

| Field | Meaning |
|---|---|
| `hz` | Inference rate: 5, 10, 25 or 50 Hz (whole multiples of the 20 ms control step). |
| `observation` | The model's input vector, in order: each entry is a joint's `pos` (position) or `vel` (velocity). Joint names are the **simulator's** names (`gr_brain.platform.robot(id).model_joints[*].name`). |
| `units` | `lerobot` — revolute joints in **degrees**, gripper **0–100 %** of travel (0 = closed), prismatic joints in **metres**, velocities per second. `sim` — radians / metres. |
| `normalize` | Optional. Applied to the input: `(x − mean) / std`, one value per input. |
| `action.joints` | The joints the output drives, in order. |
| `action.mode` | `absolute` targets, or `delta` added to the last commanded target. |
| `action.chunk` | Actions per inference (ACT-style chunking, 1–50). The output has `chunk × len(joints)` values: step 0 for every joint, then step 1, … Each step lasts one `1/hz` period. |
| `action.scale` | Output multiplier (e.g. if your model emits normalised actions). |
| `safety.maxDegPerSec` | Rate limit on every commanded joint (1–720 °/s). |

The platform always applies its **safety layer**: targets are clamped to each joint's
range and rate-limited, and a non-finite output stops the run before anything moves.

## Model I/O

* ONNX: **one** float32 input (shape `[inputs]` or `[1, inputs]`) and a float32 output with
  `chunk × len(action.joints)` values (any shape; it is read flat). One file, ≤ 16 MB.
* Live runtime: `act(observation) -> action` with the same sizes.

`gr-brain train` writes a matching adapter for you (`runs/<name>/adapter.json`); check one
with `gr_brain.adapter.validate_adapter(adapter, gr_brain.platform.robot("koch-v1-1"))`.

## Recordings

Brain recordings use LeRobot keys: `<lerobot_name>.pos` (e.g. `shoulder_pan.pos`) in the
same units, plus `x.vel`, `y.vel`, `theta.vel` for a driven base. The mapping between
LeRobot names and simulator names is in `gr_brain/data/platform.json`.
