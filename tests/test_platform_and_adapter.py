import pytest

from gr_brain.adapter import make_adapter, output_size, validate_adapter
from gr_brain.platform import failed_criteria, model_spec, platform, robot, robot_ids


def test_platform_data_names_every_v1_robot():
    ids = robot_ids()
    for rid in ("koch-v1-1", "digital-twin-arm", "lekiwi"):
        assert rid in ids
    koch = robot("koch-v1-1")
    assert koch.model_joints and all(j.kind != "wheel" for j in koch.model_joints)
    assert all(j.key.endswith(".pos") for j in koch.model_joints)


def test_stretch_alias_resolves():
    if "hello-robot-stretch-4" in robot_ids():
        assert robot("hello-stretch-4").id == "hello-robot-stretch-4"


def test_unknown_robot_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown robot"):
        robot("no-such-bot")


def test_model_spec_matches_the_platform_rules():
    doc = platform()
    spec = model_spec("abc", "koch-v1-1")
    assert spec["id"] == "model:abc" and spec["maxSimTime"] == doc["model_run_seconds"]
    assert spec["pass"] == doc["model_criteria"] and "task" not in spec
    task = robot("koch-v1-1").tasks[0]
    with_task = model_spec("abc", "koch-v1-1", 30, 2, task["id"])
    assert with_task["task"] == task and with_task["pass"][0] == doc["task_criterion"]
    assert with_task["seed"] == 2 and with_task["maxSimTime"] == 30.0


def test_failed_criteria_explain_a_fail():
    spec = model_spec("abc", "koch-v1-1")
    lines = failed_criteria(spec["pass"], {"hard_violations": 15, "worst_torque_pct": 100})
    assert len(lines) == 2 and lines[0].startswith("hard_violations = 15 (needs == 0)")
    assert failed_criteria(spec["pass"], {"hard_violations": 0, "worst_torque_pct": 40}) == []


def test_a_task_from_another_robot_is_refused():
    with pytest.raises(KeyError):
        model_spec("abc", "lekiwi", task_id="koch-peg-in-bin")


def test_valid_adapter_has_no_problems():
    koch = robot("koch-v1-1")
    names = [j.name for j in koch.model_joints]
    a = make_adapter(names, hz=25, chunk=4, mean=[0.0] * len(names), std=[1.0] * len(names))
    assert validate_adapter(a, koch) == []
    assert output_size(a) == 4 * len(names)


@pytest.mark.parametrize("mutate, expect", [
    (lambda a: a.update(hz=30), "Rate must be"),
    (lambda a: a["action"].update(joints=["nope"]), 'Output joint "nope"'),
    (lambda a: a["observation"].append({"joint": "ghost", "field": "pos"}), 'Input joint "ghost"'),
    (lambda a: a["action"].update(chunk=51), "Chunk must be"),
    (lambda a: a["safety"].update(maxDegPerSec=0), "Speed limit"),
    (lambda a: a.update(normalize={"mean": [0.0], "std": [1.0]}), "Normalisation needs"),
    (lambda a: a.update(normalize={"mean": [0.0] * 6, "std": [0.0] * 6}), "no standard deviation may be zero"),
])
def test_invalid_adapters_are_explained(mutate, expect):
    koch = robot("koch-v1-1")
    names = [j.name for j in koch.model_joints]
    a = make_adapter(names, hz=25)
    if "normalize" in str(expect) or "standard" in str(expect):
        a["observation"] = a["observation"][: len(names)]
    mutate(a)
    problems = validate_adapter(a, koch)
    assert any(expect in p for p in problems), problems
