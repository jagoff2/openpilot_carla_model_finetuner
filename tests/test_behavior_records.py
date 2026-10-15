import copy

import numpy as np
import pytest

from lane_dataset.geometry import EvidenceFrame, ObservableReference
from lane_behavior.target import ObservedRoad
from lane_behavior.records import build_record, replay_record, training_corridor


def record():
    x = np.arange(-10., 61.)
    refs = (EvidenceFrame(10, .5, "wide"),)
    ref = ObservableReference(np.c_[x, x*0-4.5, x*0], np.c_[x, x*0+4.5, x*0],
        x, [refs]*len(x), .5)
    return build_record(ObservedRoad(ref), label_frame_id=10, label_timestamp_s=.5,
        evidence_bindings={"road-proof": "a"*64}, ego_position_world=[0., 2.25, 0.], ego_yaw_rad=0.)


def test_record_replays_and_builds_ego_relative_right_lane_target():
    value = record()
    assert value["moving_supervision"]
    assert value["training_ready"] is False
    corridor = training_corridor(value, verified_evidence_bindings={"road-proof": "a"*64})
    np.testing.assert_allclose(corridor.sample(np.array([0., 10., 30.]))[0],
        [[0., 0.], [10., 0.], [30., 0.]], atol=1e-10)
    assert corridor.ego_s == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize("field", ["ego_yaw_rad", "label_frame_id", "moving_supervision"])
def test_record_changes_cannot_reuse_old_digest(field):
    value = record()
    value[field] = 23
    with pytest.raises(ValueError, match="digest mismatch"):
        replay_record(value, verified_evidence_bindings={"road-proof": "a"*64})


def test_self_consistent_hash_does_not_replace_external_artifact_binding():
    value = record()
    with pytest.raises(ValueError, match="binding mismatch"):
        replay_record(value, verified_evidence_bindings={"road-proof": "b"*64})


def test_recomputed_hash_cannot_relabel_nominal_geometry_as_an_avoidance_event():
    from lane_behavior.records import digest
    value = record()
    value["excursions"] = [{"fake": True}]
    value["record_sha256"] = digest({k: v for k, v in value.items() if k != "record_sha256"})
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        replay_record(value, verified_evidence_bindings={"road-proof": "a"*64})


def test_replaying_record_does_not_mutate_it():
    value = record()
    before = copy.deepcopy(value)
    replay_record(value, verified_evidence_bindings={"road-proof": "a"*64})
    assert value == before
