import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from gr_brain.platform import robot
from gr_brain.runtime.policies import load_policy
from gr_brain.runtime.serve import Session, relay_url, run_once

KOCH = [j.name for j in robot("koch-v1-1").model_joints]

POLICY_PY = '''
import numpy as np

class Hold:
    robot = "koch-v1-1"
    name = "hold"
    def __init__(self, joints):
        self.adapter = {"version": 1, "hz": 25, "units": "lerobot",
                        "observation": [{"joint": j, "field": "pos"} for j in joints],
                        "action": {"joints": joints, "mode": "absolute", "chunk": 1, "scale": 1.0},
                        "safety": {"maxDegPerSec": 180.0}}
        self.seeds = []
    def reset(self, seed):
        self.seeds.append(seed)
    def act(self, obs):
        return np.asarray(obs) + 1.0

def make_policy():
    return Hold(JOINTS)
'''


@pytest.fixture
def py_policy(tmp_path):
    path = tmp_path / "my_policy.py"
    path.write_text(POLICY_PY.replace("JOINTS", repr(KOCH)))
    return path


def test_the_shipped_example_policy_runs():
    example = Path(__file__).resolve().parents[1] / "examples" / "custom_policy.py"
    p = load_policy(str(example))
    s = Session(p, log=lambda _m: None)
    assert s.handle({"type": "hello", "protocol": 1, "robot": "koch-v1-1"})["type"] == "ready"
    obs = [10.0] * p.inputs
    outs = [s.handle({"type": "step", "seq": i, "obs": obs})["action"] for i in range(6)]
    assert outs[0] == obs                                   # starts by holding where it is
    assert outs[5][1] != obs[1] and outs[5][0] == obs[0]    # then sweeps joint 1 only


def test_relay_url_follows_the_api_scheme():
    assert relay_url("https://api.genuinerobotics.com") == "wss://api.genuinerobotics.com/api/v1/brain/relay/runtime"
    assert relay_url("http://localhost:8000/") == "ws://localhost:8000/api/v1/brain/relay/runtime"


def test_a_python_policy_loads_and_is_validated(py_policy):
    p = load_policy(str(py_policy))
    assert p.kind == "python" and p.robot == "koch-v1-1" and p.inputs == p.outputs == len(KOCH)


def test_a_policy_with_a_wrong_adapter_is_refused(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(POLICY_PY.replace("JOINTS", "['not_a_joint']"))
    with pytest.raises(ValueError, match="not on koch-v1-1"):
        load_policy(str(bad))


def test_session_speaks_the_protocol(py_policy):
    s = Session(load_policy(str(py_policy)), log=lambda _m: None)
    assert s.handle({"type": "attach"}) is None
    ready = s.handle({"type": "hello", "protocol": 1, "robot": "koch-v1-1", "seed": 4})
    assert ready["type"] == "ready" and ready["inputs"] == len(KOCH)
    out = s.handle({"type": "step", "seq": 3, "obs": [1.0] * len(KOCH)})
    assert out["type"] == "action" and out["seq"] == 3 and out["action"] == [2.0] * len(KOCH)
    assert s.handle({"type": "step", "seq": 4, "obs": [1.0]})["code"] == "obs"
    assert s.handle({"type": "hello", "protocol": 2})["code"] == "protocol"
    assert s.handle({"type": "hello", "protocol": 1, "robot": "lekiwi"})["code"] == "robot"
    assert s.handle({"type": "ping"}) == {"type": "pong"}


def test_a_crashing_or_non_finite_policy_reports_an_error(py_policy):
    p = load_policy(str(py_policy))
    s = Session(p, log=lambda _m: None)
    p.act = lambda _o: np.full(len(KOCH), np.nan)
    assert s.handle({"type": "step", "seq": 1, "obs": [0.0] * len(KOCH)})["code"] == "action"

    def boom(_o):
        raise RuntimeError("CUDA out of memory")
    p.act = boom
    err = s.handle({"type": "step", "seq": 2, "obs": [0.0] * len(KOCH)})
    assert err["code"] == "policy" and "out of memory" in err["message"]


def test_runtime_over_a_real_socket(py_policy):
    """A fake relay plays the platform + browser; the runtime must authenticate,
    register, answer hello and steps, and stop on a fatal close."""
    websockets = pytest.importorskip("websockets")
    seen = {}

    async def relay(ws):
        seen["auth"] = json.loads(await ws.recv())
        seen["register"] = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "registered", "runtime_id": "r1"}))
        await ws.send(json.dumps({"type": "attach"}))
        await ws.send(json.dumps({"type": "hello", "protocol": 1, "robot": "koch-v1-1", "seed": 1}))
        seen["ready"] = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "step", "seq": 1, "t": 0.0, "obs": [0.5] * len(KOCH)}))
        seen["action"] = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "error", "code": 4401, "message": "token revoked"}))
        await asyncio.sleep(0.2)

    async def main():
        async with websockets.serve(relay, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await run_once(f"ws://127.0.0.1:{port}/api/v1/brain/relay/runtime", "grb_x",
                                  load_policy(str(py_policy)), "test-pc", lambda _m: None)

    code = asyncio.run(main())
    assert code == 4401
    assert seen["auth"] == {"type": "auth", "token": "grb_x"}
    reg = seen["register"]
    assert reg["protocol"] == 1 and reg["robot"] == "koch-v1-1" and reg["adapter"]["action"]["joints"] == KOCH
    assert seen["ready"]["type"] == "ready"
    assert seen["action"] == {"type": "action", "seq": 1, "action": [1.5] * len(KOCH), "ms": seen["action"]["ms"]}


def test_an_exported_run_serves_through_onnxruntime(tmp_path, dataset_doc):
    pytest.importorskip("torch")
    from gr_brain.export import export_run
    from gr_brain.train import TrainConfig, train

    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps(dataset_doc))
    train(ds, tmp_path / "run", TrainConfig(epochs=10, hidden=32, chunk=2, device="cpu"), log=lambda _m: None)
    export_run(tmp_path / "run")
    p = load_policy(str(tmp_path / "run"))
    assert p.kind == "onnx" and p.robot == "koch-v1-1" and p.outputs == 2 * len(KOCH)
    s = Session(p, log=lambda _m: None)
    s.handle({"type": "hello", "protocol": 1, "robot": "koch-v1-1"})
    out = s.handle({"type": "step", "seq": 1, "obs": [0.0] * p.inputs})
    assert out["type"] == "action" and len(out["action"]) == 2 * len(KOCH)
    # A bare model.onnx next to its adapter.json + checkpoint loads too.
    assert load_policy(str(Path(tmp_path / "run" / "model.onnx"))).outputs == p.outputs
