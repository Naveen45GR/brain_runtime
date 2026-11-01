"""gr-brain - the command line.

    gr-brain login --token grb_...            save your access token (create it on the Brain page)
    gr-brain projects                       your projects and their robots
    gr-brain datasets <project>             recorded datasets in a project
    gr-brain pull <project> <dataset>       download a dataset (gr-brain-dataset/1 JSON)
    gr-brain train <dataset.json>           behaviour cloning on your GPU/CPU -> runs/<name>/
    gr-brain export <run>                   -> <run>/model.onnx (checked with onnxruntime)
    gr-brain upload <run> --project <id>    add the model to your project [--verify]
    gr-brain verify <model> --project <id>  re-run it on our servers -> a trusted result
    gr-brain serve <policy>                 run a policy live against the Brain simulator
    gr-brain doctor                         check this machine and your connection
"""

from __future__ import annotations

import argparse
import json
import platform as _platform
import sys
from pathlib import Path
from typing import Callable, List, Optional

from gr_brain import __version__, config
from gr_brain.client import APIError, BrainClient

ClientFactory = Callable[[config.Config], BrainClient]


def _client(cfg: config.Config) -> BrainClient:
    return BrainClient(cfg.api, cfg.token)


def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _project_robot(c: BrainClient, project_id: str, robot: Optional[str]) -> str:
    if robot:
        return robot
    p = c.project(project_id)
    if not p.get("robot"):
        raise APIError(422, f"project {p['name']!r} has no Brain robot - pass --robot")
    return p["robot"]


# -- commands ----------------------------------------------------------------

def cmd_login(a, make: ClientFactory) -> int:
    cfg = config.load()
    if a.api:
        cfg.api = a.api.rstrip("/")
    cfg.token = a.token.strip()
    if not cfg.token.startswith("grb_"):
        _say("warning: Brain access tokens start with grb_ - create one on the Brain page (Train on your own GPU).")
    with make(cfg) as c:
        projects = c.projects()
    path = config.save(cfg)
    _say(f"signed in to {cfg.api} - {len(projects)} project(s) visible. Saved to {path}")
    return 0


def cmd_projects(a, make: ClientFactory) -> int:
    with make(config.load()) as c:
        rows = c.projects()
    if a.json:
        _say(json.dumps(rows, indent=2))
        return 0
    if not rows:
        _say("no projects yet - create one on the platform.")
    for p in rows:
        _say(f"{p['id']}  {(p.get('robot') or '-'):<24} {p['name']}{'' if p.get('mine', True) else '  (team)'}")
    return 0


def cmd_datasets(a, make: ClientFactory) -> int:
    with make(config.load()) as c:
        robot = _project_robot(c, a.project, a.robot)
        eps = c.episodes(a.project, robot)
    groups: dict = {}
    for e in eps:
        g = groups.setdefault(e["dataset"], {"episodes": 0, "frames": 0, "seconds": 0.0})
        g["episodes"] += 1
        g["frames"] += e.get("frames", 0)
        g["seconds"] += e.get("duration_s", 0.0)
    if not groups:
        _say(f"no recordings for {robot} in this project - record some in Brain -> Control -> Record.")
    for name, g in sorted(groups.items()):
        _say(f"{name:<32} {g['episodes']:4d} episodes  {g['frames']:7d} frames  {g['seconds']:7.1f} s")
    return 0


def cmd_pull(a, make: ClientFactory) -> int:
    with make(config.load()) as c:
        robot = _project_robot(c, a.project, a.robot)
        doc = c.dataset(a.project, a.dataset, robot)
    out = Path(a.output or f"{robot}_{a.dataset}.json")
    out.write_text(json.dumps(doc), encoding="utf-8")
    n = len(doc.get("episodes") or [])
    _say(f"saved {n} episode(s) of {robot} -> {out}")
    return 0


def cmd_train(a, _make: ClientFactory) -> int:
    from gr_brain.train import TrainConfig, train

    cfg = TrainConfig(hz=a.hz, chunk=a.chunk, hidden=a.hidden, layers=a.layers, epochs=a.epochs,
                      lr=a.lr, seed=a.seed, device=a.device, max_deg_per_sec=a.max_speed)
    name = a.name or Path(a.dataset).stem
    train(a.dataset, Path(a.runs) / name, cfg, log=_say)
    return 0


def cmd_export(a, _make: ClientFactory) -> int:
    from gr_brain.export import export_run

    r = export_run(a.run)
    _say(f"exported {r.onnx_path} ({r.size_bytes / 1024:.0f} KB) - onnxruntime matches PyTorch within {r.max_abs_diff:.1e}")
    return 0


def _verify(c: BrainClient, project: str, robot: str, model_id: str, seeds: int, task: Optional[str],
            seconds: Optional[float]) -> int:
    from gr_brain.platform import model_spec, platform

    limits = platform()["verify"]
    if not 1 <= seeds <= limits["max_seeds"]:
        raise APIError(422, f"--seeds must be 1-{limits['max_seeds']}")
    spec = model_spec(model_id, robot, seconds, 1, task)
    if spec["maxSimTime"] > limits["max_sim_seconds"]:
        raise APIError(422, f"--seconds must be <= {limits['max_sim_seconds']}")
    body = {"kind": "model", "robot": robot, "behaviour_id": f"model:{model_id}", "spec": spec,
            "seeds": list(range(1, seeds + 1)), "scene": None, "model_id": model_id, "claimed": []}
    v = c.request_verification(project, body)
    _say(f"verifying on our servers ({seeds} seed(s), up to {v['estimated_credits']} credits)...")
    v = c.wait_verification(v["id"])
    if v["status"] == "failed":
        _say(f"verification failed: {v.get('error') or 'unknown error'}")
        return 1
    res = v["result"]
    _say(f"{(v.get('trust_level') or '').replace('_', ' ')}: {res['passed']}/{res['total']} PASS "
         f"| {res['sim_seconds']} s simulated | {v.get('credits')} credits")
    from gr_brain.platform import failed_criteria
    for run in res.get("runs", []):
        reason = f" - {run['reason']}" if run.get("reason") else ""
        _say(f"  seed {run['seed']}: {run['status']}{reason}")
        if run.get("status") == "FAIL":
            for line in failed_criteria(spec["pass"], run.get("measures") or {}):
                _say(f"      {line}")
    return 0 if res["passed"] == res["total"] else 2


def cmd_upload(a, make: ClientFactory) -> int:
    run = Path(a.run)
    onnx = run / "model.onnx"
    if not onnx.exists():
        raise APIError(422, f"{onnx} not found - run `gr-brain export {run}` first")
    adapter = json.loads((run / "adapter.json").read_text(encoding="utf-8"))
    info = json.loads((run / "model_info.json").read_text(encoding="utf-8"))
    from gr_brain.train import load_checkpoint
    robot = load_checkpoint(run)["robot"]
    with make(config.load()) as c:
        project_robot = c.project(a.project).get("robot")
        if project_robot and project_robot != robot:
            raise APIError(422, f"this model drives {robot}, but the project's robot is {project_robot}")
        m = c.upload_model(a.project, robot=robot, name=a.name or run.name, onnx_bytes=onnx.read_bytes(),
                           adapter=adapter, info=info, licence=a.licence or "")
        _say(f"uploaded model {m['id']} ({m['name']}) - it is in Brain -> your models.")
        if a.verify:
            return _verify(c, a.project, robot, m["id"], a.seeds, a.task, a.seconds)
    return 0


def cmd_verify(a, make: ClientFactory) -> int:
    with make(config.load()) as c:
        robot = _project_robot(c, a.project, a.robot)
        return _verify(c, a.project, robot, a.model, a.seeds, a.task, a.seconds)


def cmd_serve(a, _make: ClientFactory) -> int:
    from gr_brain.runtime.serve import serve

    return serve(policy=a.policy, name=a.name, api=a.api, device=a.device, robot=a.robot, adapter=a.adapter, log=_say)


def cmd_doctor(a, _make: ClientFactory) -> int:
    ok = True
    _say(f"gr-brain {__version__} | Python {_platform.python_version()} | {_platform.system()} {_platform.machine()}")
    try:
        import onnxruntime as ort
        _say(f"onnxruntime {ort.__version__} | providers: {', '.join(ort.get_available_providers())}")
    except ImportError:
        ok = False
        _say("onnxruntime: MISSING - pip install onnxruntime (or onnxruntime-gpu)")
    try:
        import torch
        dev = "cuda " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu only")
        _say(f"torch {torch.__version__} | {dev}")
    except ImportError:
        _say("torch: not installed (needed for train/export: pip install 'gr-brain[train]')")
    cfg = config.load()
    _say(f"api: {cfg.api}")
    if not cfg.token:
        ok = False
        _say("token: not set - create one on the Brain page, then `gr-brain login --token ...`")
    else:
        try:
            with BrainClient(cfg.api, cfg.token) as c:
                _say(f"token: OK - {len(c.projects())} project(s) visible")
        except APIError as exc:
            ok = False
            _say(f"token: {exc}")
    return 0 if ok else 1


# -- parser ------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gr-brain", description="Train, upload and run policies for Genuine Robotics Brain.")
    p.add_argument("--version", action="version", version=f"gr-brain {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("login", help="save your access token")
    s.add_argument("--token", required=True)
    s.add_argument("--api", help=f"API address (default {config.DEFAULT_API})")
    s.set_defaults(fn=cmd_login)

    s = sub.add_parser("projects", help="your projects and their robots")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_projects)

    s = sub.add_parser("datasets", help="recorded datasets in a project")
    s.add_argument("project")
    s.add_argument("--robot")
    s.set_defaults(fn=cmd_datasets)

    s = sub.add_parser("pull", help="download a dataset")
    s.add_argument("project")
    s.add_argument("dataset")
    s.add_argument("--robot")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_pull)

    s = sub.add_parser("train", help="behaviour cloning on a downloaded dataset")
    s.add_argument("dataset")
    s.add_argument("--name")
    s.add_argument("--runs", default="runs")
    s.add_argument("--hz", type=int, choices=[5, 10, 25, 50])
    s.add_argument("--chunk", type=int, default=1)
    s.add_argument("--hidden", type=int, default=256)
    s.add_argument("--layers", type=int, default=2)
    s.add_argument("--epochs", type=int, default=400)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", choices=["cuda", "mps", "cpu"])
    s.add_argument("--max-speed", type=float, default=180.0, help="adapter safety limit, deg/s")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("export", help="training run -> model.onnx")
    s.add_argument("run")
    s.set_defaults(fn=cmd_export)

    def verify_opts(s):
        s.add_argument("--seeds", type=int, default=3)
        s.add_argument("--task", help="judge against a platform task (e.g. koch-peg-in-bin)")
        s.add_argument("--seconds", type=float, help="simulated seconds per run")

    s = sub.add_parser("upload", help="add an exported model to a project")
    s.add_argument("run")
    s.add_argument("--project", required=True)
    s.add_argument("--name")
    s.add_argument("--licence", default="Apache-2.0")
    s.add_argument("--verify", action="store_true", help="then verify it on our servers")
    verify_opts(s)
    s.set_defaults(fn=cmd_upload)

    s = sub.add_parser("verify", help="re-run an uploaded model on our servers")
    s.add_argument("model")
    s.add_argument("--project", required=True)
    s.add_argument("--robot")
    verify_opts(s)
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("serve", help="run a policy live against the Brain simulator")
    s.add_argument("policy", help="model.onnx, a training run folder, or my_policy.py")
    s.add_argument("--name", help="how this runtime appears in Brain")
    s.add_argument("--api", help="API address (default: your login's)")
    s.add_argument("--device", help="cuda | cpu (default: best available)")
    s.add_argument("--robot", help="for a model.onnx that did not come from `gr-brain train`")
    s.add_argument("--adapter", help="adapter.json for that model (default: next to it)")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("doctor", help="check this machine and your connection")
    s.set_defaults(fn=cmd_doctor)
    return p


def main(argv: Optional[List[str]] = None, make: ClientFactory = _client) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.fn(args, make) or 0)
    except APIError as exc:
        _say(f"error: {exc.detail if exc.status else exc}")
        return 1
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        _say(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
