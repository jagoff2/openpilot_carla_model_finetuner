from copy import deepcopy

import numpy as np
import pytest

from lane_dataset.admission import validate_recovery_event
from lane_dataset.geometry import LaneReference
from lane_dataset.quality import qualify_episode, recovery_envelope


def recovery_fixture(sign=1.):
    t = np.arange(201)*.05
    offset = sign*np.where(t < 5., .25, np.maximum(0., .25*(1-(t-5.)/2.)))
    arc = np.arange(61, dtype=float)
    center = np.c_[arc, np.zeros(len(arc)), np.zeros(len(arc))]
    reference = LaneReference(center-[0., 1.8, 0.], center+[0., 1.8, 0.], arc)
    telemetry = dict(timestamp=t, position_world=np.c_[5.+4*t, offset, np.zeros(len(t))])
    manifest = dict(recovery=True, recovery_onset_s=5.,
        recovery_event=dict(onset_index=100, onset_s=5., held_offset_m=sign*.25,
                            physical_offset_at_release_m=sign*.25),
        label_records=[dict(label_index=100), dict(label_index=130)])
    return reference, telemetry, manifest


@pytest.mark.parametrize("sign", [-1., 1.])
def test_signed_observed_recovery_remains_valid(sign):
    ref, telemetry, manifest = recovery_fixture(sign)
    report = validate_recovery_event(manifest, telemetry, ref)
    assert report["passed"], report["errors"]
    envelope = recovery_envelope(telemetry["timestamp"], telemetry["position_world"][:, 1], 5.)
    assert envelope["passed"], envelope["errors"]


def test_recovery_prefix_cannot_hide_an_unrelated_center_excursion():
    ref, telemetry, _ = recovery_fixture()
    # This returns to the declared release position and converges normally, so
    # the old recovery waiver discarded every sample revealing the excursion.
    telemetry["position_world"][20:60, 1] = -.25
    t = telemetry["timestamp"]
    result = qualify_episode(timestamps_s=t, positions_xyz=telemetry["position_world"],
        yaw_rad=np.zeros(len(t)), speeds_mps=np.full(len(t), 4.),
        velocity_world=np.c_[np.full(len(t), 4.), np.zeros((len(t), 2))],
        lateral_acceleration_mps2=np.zeros(len(t)), steering_fractions=np.zeros(len(t)),
        reference=ref, vehicle_length_m=4.6, vehicle_width_m=1.85,
        collisions=0, rendered_boundary_agreement=True, completed_maneuver=True,
        intended_progress_m=39.9, recovery=True, recovery_onset_s=5.)
    assert not result.accepted
    assert "recovery_prefix_does_not_hold_initial_offset" in result.reasons


@pytest.mark.parametrize("bad_offset,reason", [
    (.40, "recovery_moves_farther_from_center"),
    (-.20, "recovery_overshoot_above_limit"),
])
def test_converging_later_does_not_excuse_large_recovery_excursion(bad_offset, reason):
    _, telemetry, _ = recovery_fixture()
    offset = telemetry["position_world"][:, 1].copy()
    offset[110] = bad_offset
    report = recovery_envelope(telemetry["timestamp"], offset, 5.)
    assert reason in report["errors"]


def test_recovery_metadata_cannot_change_side_or_invent_its_release():
    reference, telemetry, manifest = recovery_fixture()
    wrong_side = deepcopy(manifest)
    wrong_side["recovery_event"]["held_offset_m"] = -.25
    assert "measured_recovery_offset_disagrees_with_signed_hold" in validate_recovery_event(
        wrong_side, telemetry, reference)["errors"]
    changed_measurement = deepcopy(manifest)
    changed_measurement["recovery_event"]["physical_offset_at_release_m"] = .21
    assert "measured_recovery_release_offset_mismatch" in validate_recovery_event(
        changed_measurement, telemetry, reference)["errors"]
    changed_time = deepcopy(manifest)
    changed_time["recovery_event"]["onset_s"] = 5.1
    assert "measured_recovery_onset_timing_mismatch" in validate_recovery_event(
        changed_time, telemetry, reference)["errors"]


def test_held_offset_prefix_cannot_supply_training_labels():
    reference, telemetry, manifest = recovery_fixture()
    manifest["label_records"].append(dict(label_index=90))
    report = validate_recovery_event(manifest, telemetry, reference)
    assert "label_precedes_declared_recovery_release" in report["errors"]


def test_recovery_without_declaration_cannot_claim_a_delayed_release():
    reference, telemetry, manifest = recovery_fixture()
    manifest["recovery"] = False
    report = validate_recovery_event(manifest, telemetry, reference)
    assert not report["passed"]
    assert report["errors"][0].startswith("recovery_event_evidence_invalid:")


def test_recovery_release_must_be_an_observed_frame():
    _, telemetry, _ = recovery_fixture()
    report = recovery_envelope(telemetry["timestamp"], telemetry["position_world"][:, 1], 5.025)
    assert "recovery_onset_not_recorded_frame" in report["errors"]
