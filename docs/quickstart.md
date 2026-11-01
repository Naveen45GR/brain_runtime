# Quickstart — from demonstrations to a verified model

This walks the Koch v1.1 arm through the whole loop. Any Brain robot works the same way.

## 1. Record demonstrations (in the browser)

1. Open your Koch project on the platform → **Brain** → **Control**.
2. Name a dataset (e.g. `pick-demos`), press **Record**, drive the arm with the sliders or
   keys through the task, press **Stop**, then **Save**. Repeat 10–50 times, varying how
   you do it a little each time.

## 2. Install and sign in

```bash
pip install "gr-brain[train]"
```

Brain → **Train on your own GPU** → *Create token* → copy it, then:

```bash
gr-brain login --token grb_…
gr-brain doctor
```

## 3. Pull and train

```bash
gr-brain projects                          # find the project id
gr-brain datasets <project-id>
gr-brain pull <project-id> pick-demos      # → koch-v1-1_pick-demos.json
gr-brain train koch-v1-1_pick-demos.json --name pick --chunk 4
```

The validation error printed at the end is in the recording's units (degrees for most
joints). If it is large, record more (and more consistent) demonstrations.

## 4. Export, upload, verify

```bash
gr-brain export runs/pick
gr-brain upload runs/pick --project <project-id> --verify --task koch-peg-in-bin --seconds 30
```

You will see something like:

```
uploaded model 6f1c… (pick) - it is in Brain -> your models.
verifying on our servers (3 seed(s), up to 90 credits)...
model verified: 2/3 PASS | 61.2 s simulated | 62 credits
  seed 1: PASS
  seed 2: PASS
  seed 3: FAIL
      task.success = 0 (needs == 1) - The task is judged from where things end up.
```

In Brain, the model is under **Your models**: *Run* it to watch it, *Evaluate* it over more
seeds, *Verify on our servers* again any time.

## 5. (Optional) Run it live from your GPU

```bash
gr-brain serve runs/pick --name "lab workstation"
```

In Brain, **lab workstation** appears as *online* — run and evaluate it like any brain.
This is how to try a model that is too big for the browser (a VLA, a diffusion policy):
wrap it in a `my_policy.py` (see `examples/custom_policy.py`) and `gr-brain serve my_policy.py`.
