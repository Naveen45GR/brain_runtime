import base64
import json

import httpx
import pytest

from gr_brain import cli, config
from gr_brain.client import APIError, BrainClient

PROJECT = "11111111-1111-1111-1111-111111111111"


class FakeAPI:
    """The Brain endpoints gr-brain calls, in memory."""

    def __init__(self, dataset_doc=None):
        self.calls = []
        self.models = {}
        self.dataset_doc = dataset_doc
        self.polls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.headers.get("authorization") != "Bearer grb_test":
            return httpx.Response(401, json={"detail": "This Brain access token is invalid, expired or revoked."})
        p, m = request.url.path, request.method
        if p == "/api/v1/brain/projects":
            return httpx.Response(200, json=[{"id": PROJECT, "name": "Koch lab", "robot": "koch-v1-1", "mine": True}])
        if p.endswith("/brain/episodes"):
            return httpx.Response(200, json=[{"dataset": "demo", "frames": 120, "duration_s": 4.0}] * 2)
        if "/datasets/" in p:
            return httpx.Response(200, json=self.dataset_doc)
        if p.endswith("/brain/models") and m == "POST":
            body = json.loads(request.content)
            self.models["m1"] = body
            return httpx.Response(201, json={"id": "m1", "name": body["name"]})
        if p.endswith("/brain/verifications") and m == "POST":
            self.verification_body = json.loads(request.content)
            return httpx.Response(202, json={"id": "v1", "status": "queued", "estimated_credits": 30})
        if p == "/api/v1/brain/verifications/v1":
            self.polls += 1
            if self.polls < 2:
                return httpx.Response(200, json={"id": "v1", "status": "running"})
            return httpx.Response(200, json={
                "id": "v1", "status": "done", "trust_level": "model_verified", "credits": 30,
                "result": {"passed": 3, "total": 3, "sim_seconds": 30.0,
                           "runs": [{"seed": s, "status": "PASS"} for s in (1, 2, 3)]}})
        return httpx.Response(404, json={"detail": "not found"})


def factory(api: FakeAPI):
    def make(cfg):
        return BrainClient(cfg.api, cfg.token, transport=httpx.MockTransport(api))
    return make


def test_no_token_is_a_helpful_error():
    with pytest.raises(APIError, match="gr-brain login"):
        BrainClient("https://x", None)


def test_errors_carry_the_servers_detail():
    api = FakeAPI()
    c = BrainClient("https://x", "grb_wrong", transport=httpx.MockTransport(api))
    with pytest.raises(APIError) as e:
        c.projects()
    assert e.value.status == 401 and "revoked" in e.value.detail


def test_login_checks_the_token_then_saves_it(capsys):
    api = FakeAPI()
    assert cli.main(["login", "--token", "grb_test", "--api", "https://api.example"], make=factory(api)) == 0
    cfg = config.load()
    assert cfg.token == "grb_test" and cfg.api == "https://api.example"
    assert "1 project(s)" in capsys.readouterr().out


def test_login_with_a_bad_token_saves_nothing():
    assert cli.main(["login", "--token", "grb_wrong"], make=factory(FakeAPI())) == 1
    assert config.load().token is None


def test_projects_datasets_and_pull(tmp_path, capsys, dataset_doc):
    config.save(config.Config(api="https://api.example", token="grb_test"))
    api = FakeAPI(dataset_doc)
    assert cli.main(["projects"], make=factory(api)) == 0
    assert "koch-v1-1" in capsys.readouterr().out
    assert cli.main(["datasets", PROJECT], make=factory(api)) == 0
    assert "demo" in capsys.readouterr().out
    out = tmp_path / "d.json"
    assert cli.main(["pull", PROJECT, "demo", "-o", str(out)], make=factory(api)) == 0
    assert json.loads(out.read_text())["format"] == "gr-brain-dataset/1"


def test_upload_and_verify(tmp_path, capsys, dataset_doc, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from gr_brain.export import export_run
    from gr_brain.train import TrainConfig, train

    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps(dataset_doc))
    train(ds, tmp_path / "run", TrainConfig(epochs=20, hidden=32, device="cpu"), log=lambda _m: None)
    export_run(tmp_path / "run")
    config.save(config.Config(api="https://api.example", token="grb_test"))
    api = FakeAPI()
    import gr_brain.client as client_mod
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)
    code = cli.main(["upload", str(tmp_path / "run"), "--project", PROJECT, "--name", "wave",
                     "--verify", "--seeds", "3", "--task", "koch-peg-in-bin", "--seconds", "20"], make=factory(api))
    out = capsys.readouterr().out
    assert code == 0, out
    body = api.models["m1"]
    assert body["robot"] == "koch-v1-1" and body["name"] == "wave"
    assert base64.b64decode(body["content_base64"])[:1] == b"\x08"     # an ONNX protobuf
    assert body["info"]["inputName"] == "observation"
    v = api.verification_body
    assert v["kind"] == "model" and v["model_id"] == "m1" and v["seeds"] == [1, 2, 3]
    assert v["spec"]["task"]["id"] == "koch-peg-in-bin" and v["spec"]["maxSimTime"] == 20.0
    assert "model verified: 3/3 PASS" in out


def test_upload_refuses_a_model_for_another_robot(tmp_path, dataset_doc):
    pytest.importorskip("torch")
    from conftest import synthetic_dataset
    from gr_brain.export import export_run
    from gr_brain.train import TrainConfig, train

    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps(synthetic_dataset(robot_id="digital-twin-arm")))
    train(ds, tmp_path / "run", TrainConfig(epochs=5, hidden=16, device="cpu"), log=lambda _m: None)
    export_run(tmp_path / "run")
    config.save(config.Config(api="https://api.example", token="grb_test"))
    assert cli.main(["upload", str(tmp_path / "run"), "--project", PROJECT], make=factory(FakeAPI())) == 1


def test_verify_limits_are_checked_before_calling(capsys):
    config.save(config.Config(api="https://api.example", token="grb_test"))
    api = FakeAPI()
    assert cli.main(["verify", "m1", "--project", PROJECT, "--seeds", "11"], make=factory(api)) == 1
    assert not any(c[1].endswith("/verifications") for c in api.calls)
