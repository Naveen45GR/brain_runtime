"""``gr-brain serve`` - connect a policy to the Brain simulator through the platform relay.

    runtime -> relay   {"type":"auth","token"}            then
                      {"type":"register","protocol":1,"name","robot","device","adapter","model","version"}
    relay   -> runtime {"type":"registered","runtime_id"}
    ... a Brain page attaches ...
    relay   -> runtime {"type":"attach"}
    browser -> runtime {"type":"hello","protocol":1,"robot","seed"}
    runtime -> browser {"type":"ready","inputs","outputs","device","model"}   (or "error")
    browser -> runtime {"type":"step","seq","t","obs":[...]}
    runtime -> browser {"type":"action","seq","action":[...],"ms"}
    browser -> runtime {"type":"reset","seed"} | {"type":"bye"}
    relay   -> runtime {"type":"detach"}

Full description: docs/protocol.md.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import socket
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from gr_brain import __version__, config
from gr_brain.runtime.policies import LivePolicy, load_policy

PROTOCOL = 1
RELAY_PATH = "/api/v1/brain/relay/runtime"
FATAL_CODES = {4400, 4401, 4409}     # bad request / bad token / too many runtimes: retrying won't help


def relay_url(api: str) -> str:
    base = api.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return base + RELAY_PATH


class Session:
    """One runtime's protocol handling, independent of the socket (testable)."""

    def __init__(self, policy: LivePolicy, log: Callable[[str], None]):
        self.policy = policy
        self.log = log
        self.steps = 0
        self.total_ms = 0.0

    def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        kind = msg.get("type")
        if kind == "attach":
            self.steps, self.total_ms = 0, 0.0
            self.log("a Brain page attached")
            return None
        if kind == "hello":
            if msg.get("protocol") != PROTOCOL:
                return {"type": "error", "code": "protocol", "message": f"runtime speaks protocol {PROTOCOL}"}
            if msg.get("robot") and msg["robot"] != self.policy.robot:
                return {"type": "error", "code": "robot",
                        "message": f"this runtime drives {self.policy.robot}, not {msg['robot']}"}
            self.policy.reset(int(msg.get("seed") or 1))
            return {"type": "ready", "inputs": self.policy.inputs, "outputs": self.policy.outputs,
                    "device": self.policy.device, "model": self.policy.name}
        if kind == "step":
            seq = msg.get("seq")
            obs = msg.get("obs")
            if not isinstance(obs, list) or len(obs) != self.policy.inputs:
                return {"type": "error", "seq": seq, "code": "obs",
                        "message": f"expected {self.policy.inputs} inputs, got {len(obs) if isinstance(obs, list) else 'none'}"}
            t0 = time.perf_counter()
            try:
                action = self.policy.act(np.asarray(obs, dtype=np.float32))
            except Exception as exc:  # noqa: BLE001 - the user's model; report, keep serving
                return {"type": "error", "seq": seq, "code": "policy", "message": f"{type(exc).__name__}: {exc}"[:300]}
            ms = (time.perf_counter() - t0) * 1000
            values = [float(v) for v in np.asarray(action).reshape(-1)]
            if len(values) != self.policy.outputs or not all(math.isfinite(v) for v in values):
                return {"type": "error", "seq": seq, "code": "action",
                        "message": f"policy returned {len(values)} values (expected {self.policy.outputs}) or a non-finite value"}
            self.steps += 1
            self.total_ms += ms
            return {"type": "action", "seq": seq, "action": values, "ms": round(ms, 3)}
        if kind == "reset":
            self.policy.reset(int(msg.get("seed") or 1))
            return None
        if kind in ("bye", "detach"):
            if self.steps:
                self.log(f"run ended: {self.steps} steps, {self.total_ms / self.steps:.1f} ms per inference")
                self.steps, self.total_ms = 0, 0.0
            if kind == "detach":
                self.log("the Brain page detached - waiting for the next one")
            return None
        if kind == "ping":
            return {"type": "pong"}
        if kind == "error":
            self.log(f"relay: {msg.get('message')}")
        return None


async def run_once(url: str, token: str, policy: LivePolicy, name: str, log: Callable[[str], None]) -> Optional[int]:
    """One connection. Returns a close code that means "stop", else None (reconnect)."""
    import websockets

    async with websockets.connect(url, max_size=1 << 20, ping_interval=20, ping_timeout=20,
                                  user_agent_header=f"gr-brain/{__version__}") as ws:
        await ws.send(json.dumps({"type": "auth", "token": token}))
        await ws.send(json.dumps({
            "type": "register", "protocol": PROTOCOL, "name": name, "robot": policy.robot, "device": policy.device,
            "adapter": policy.adapter, "version": __version__,
            "model": {"name": policy.name, "kind": policy.kind, "inputs": policy.inputs, "outputs": policy.outputs},
        }))
        first = json.loads(await ws.recv())
        if first.get("type") != "registered":
            log(f"the platform refused this runtime: {first.get('message') or first}")
            return int(first.get("code") or 4400)
        log(f"online as '{name}' ({policy.robot}, {policy.device}). In Brain, pick '{name}' under Local runtimes.")
        session = Session(policy, log)
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "error" and msg.get("code") in FATAL_CODES:
                log(f"the platform closed this runtime: {msg.get('message')}")
                return int(msg["code"])
            reply = session.handle(msg)
            if reply is not None:
                await ws.send(json.dumps(reply))
    return None


async def run_forever(url: str, token: str, policy: LivePolicy, name: str, log: Callable[[str], None]) -> int:
    delay = 1.0
    while True:
        started = time.monotonic()
        try:
            code = await run_once(url, token, policy, name, log)
            if code in FATAL_CODES:
                return 1
        except (OSError, asyncio.TimeoutError) as exc:
            log(f"cannot reach the platform ({exc}); retrying in {delay:.0f} s")
        except Exception as exc:  # noqa: BLE001 - websockets.ConnectionClosed and friends
            log(f"connection lost ({type(exc).__name__}); reconnecting in {delay:.0f} s")
        if time.monotonic() - started > 30:
            delay = 1.0
        await asyncio.sleep(delay + random.random() * 0.5)
        delay = min(30.0, delay * 2)


def serve(policy: str, *, name: Optional[str] = None, api: Optional[str] = None, device: Optional[str] = None,
          robot: Optional[str] = None, adapter: Optional[str] = None, log: Callable[[str], None] = print) -> int:
    cfg = config.load()
    token = cfg.token
    if not token:
        log("not signed in - create a token on the Brain page, then `gr-brain login --token ...`")
        return 1
    p = load_policy(policy, robot=robot, adapter=adapter, device=device)
    label = name or f"{socket.gethostname()} | {p.name}"
    log(f"loaded {p.kind} policy {p.name!r}: {p.inputs} inputs -> {p.outputs} outputs on {p.device}")
    try:
        return asyncio.run(run_forever(relay_url(api or cfg.api), token, p, label[:60], log))
    except KeyboardInterrupt:
        log("stopped")
        return 0
