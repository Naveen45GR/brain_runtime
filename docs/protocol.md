# Live runtime protocol (v1)

How a policy running on your machine (`gr-brain serve`) drives the Brain simulator.

## Topology

Browsers don't let a website open a socket to `localhost` (Chromium's *Local Network
Access* blocks it), so both ends connect **out** to the platform relay, which pairs them:

```
gr-brain serve ──wss──► wss://api.genuinerobotics.com/api/v1/brain/relay/runtime ┐
                                                                                  ├─ same account only
Brain page     ──wss──► wss://api.genuinerobotics.com/api/v1/brain/relay/browser ┘
```

The relay forwards the messages below between two connections of the **same account**,
never reads or runs the model, caps every message at 64 KB and 200 messages per second,
and lets one Brain tab use a runtime at a time. An account may have 3 runtimes online.

All messages are JSON text frames with a `type`.

## Runtime ⇄ relay

```jsonc
// 1. first message: authenticate (a Brain access token, grb_…)
{"type": "auth", "token": "grb_…"}

// 2. second message: what this runtime is
{"type": "register", "protocol": 1,
 "name": "lab-pc · pick",            // shown in Brain (≤ 60 chars)
 "robot": "koch-v1-1",               // a Brain robot id
 "device": "cuda",                   // shown in Brain
 "adapter": { … },                   // docs/adapter.md — REQUIRED
 "model": {"name": "pick", "kind": "onnx", "inputs": 6, "outputs": 12},
 "version": "0.1.0"}

// relay → runtime
{"type": "registered", "runtime_id": "…", "protocol": 1}
{"type": "error", "code": 4401, "message": "…"}   // then the socket closes
```

Close/error codes: `4400` bad message or protocol, `4401` authentication, `4404` runtime not
online (or not yours), `4408` idle, `4409` busy / too many runtimes, `4410` runtime went away,
`4413` message too large, `4429` too many messages. `gr-brain serve` stops on 4400/4401/4409
and reconnects (with back-off) on anything else.

## A session (Brain page ⇄ runtime)

```jsonc
relay   → runtime  {"type": "attach"}                                  // a Brain page attached
browser → runtime  {"type": "hello", "protocol": 1, "robot": "koch-v1-1", "seed": 1}
runtime → browser  {"type": "ready", "inputs": 6, "outputs": 12, "device": "cuda", "model": "pick"}

// once per inference, in lockstep (the simulation waits for the answer)
browser → runtime  {"type": "step", "seq": 17, "obs": [ … inputs floats … ]}
runtime → browser  {"type": "action", "seq": 17, "action": [ … outputs floats … ], "ms": 3.2}

browser → runtime  {"type": "reset", "seed": 2}       // optional
browser → runtime  {"type": "bye"}                    // run over
relay   → runtime  {"type": "detach"}                 // the page left
relay   → browser  {"type": "runtime_gone"}           // the runtime left
```

* `obs` is built by the platform from the adapter (positions/velocities in the adapter's
  units, normalised if the adapter says so). `action` must have exactly
  `chunk × len(action.joints)` finite values — anything else is an error and nothing moves.
* A step that is not answered within **5 s** ends the run with an error.
* A runtime may answer a step with `{"type": "error", "seq": 17, "code": "policy", "message": "…"}`.
* `hello.robot` must match the runtime's robot; `ready.inputs/outputs` must match the adapter.

## Trust

Runs through a live runtime are **self-reported** (the platform can't re-run your machine).
For a verified result, export the model to ONNX and `gr-brain upload --verify`.
