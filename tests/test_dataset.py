import numpy as np
import pytest

from gr_brain.dataset import default_hz, load, to_arrays, usable_joints
from gr_brain.platform import robot


def test_load_and_joint_order(dataset_doc):
    ds = load(dataset_doc)
    assert ds.robot.id == "koch-v1-1" and len(ds.episodes) == 4
    assert [j.name for j in usable_joints(ds)] == [j.name for j in robot("koch-v1-1").model_joints]


def test_wrong_format_is_refused():
    with pytest.raises(ValueError, match="not a Brain dataset"):
        load({"format": "something-else"})


def test_resampling_holds_the_latest_frame(dataset_doc):
    ds = load(dataset_doc)
    a = to_arrays(ds, hz=10, chunk=1)
    ep0 = ds.episodes[0]
    first_key = a.joints[0].key
    # Step k at 10 Hz is t = k/10 -> frame round(k*3) of the 30 fps recording.
    for k in (0, 1, 7):
        assert a.X[k, 0] == pytest.approx(ep0.observation[k * 3][first_key], abs=1e-4)
    assert a.X.shape[1] == len(a.joints) and len(a.X) == len(a.Y) == len(a.episode)


def test_chunks_are_the_next_actions_and_hold_at_the_end(dataset_doc):
    ds = load(dataset_doc)
    a = to_arrays(ds, hz=10, chunk=3)
    J = len(a.joints)
    first = a.episode == 0
    Y = a.Y[first]
    ep0 = ds.episodes[0]
    key = a.joints[0].key
    assert Y[0, 0] == pytest.approx(ep0.action[0][key], abs=1e-4)
    assert Y[0, J] == pytest.approx(ep0.action[3][key], abs=1e-4)       # next step
    assert Y[0, 2 * J] == pytest.approx(ep0.action[6][key], abs=1e-4)
    assert np.allclose(Y[-1, :J], Y[-1, J:2 * J]) and np.allclose(Y[-1, :J], Y[-1, 2 * J:])


def test_a_joint_missing_from_some_frames_is_left_out(dataset_doc):
    key = robot("koch-v1-1").model_joints[0].key
    del dataset_doc["episodes"][1]["frames"][5]["observation"][key]
    ds = load(dataset_doc)
    assert key not in [j.key for j in usable_joints(ds)]


def test_default_rate_never_exceeds_the_recording():
    assert default_hz(30) == 25 and default_hz(12) == 10 and default_hz(60) == 25 and default_hz(3) == 5


def test_bad_rate_or_chunk_is_refused(dataset_doc):
    ds = load(dataset_doc)
    with pytest.raises(ValueError):
        to_arrays(ds, hz=30)
    with pytest.raises(ValueError):
        to_arrays(ds, chunk=0)
