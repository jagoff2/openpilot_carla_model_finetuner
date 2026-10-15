from dataclasses import replace

import numpy as np
import pytest

from lane_dataset.geometry import EvidenceFrame, LaneReference, ObservableReference
from lane_behavior.target import BehaviorTarget, ObservedRoad, build_behavior_target
from lane_behavior.window import SCOPE, WindowTarget, build_behavior_window
from test_behavior_target import box, road


def observed_slice(first=0., last=80., *, width=12.):
    original = road(width=width).reference
    take = (original.s >= first) & (original.s <= last)
    return ObservedRoad(ObservableReference(original.left_xyz[take], original.right_xyz[take],
        original.s[take], [row for row, valid in zip(original.evidence_frames, take) if valid],
        original.decision_time_s))


def test_window_is_distinct_observed_contract_with_future_only_queries():
    target = build_behavior_window(road(), source_ego_s=20.)
    assert isinstance(target, WindowTarget) and not isinstance(target, BehaviorTarget)
    assert target.moving_supervision and not target.complete_event_qualified
    assert target.scope == SCOPE
    assert target.s[0] == target.source_ego_s == 20.
    np.testing.assert_allclose(target.xy[:, 1], 3.)
    q = np.array([19., 20., 40., 80., np.nan])
    xy, _, _, _, valid = target.sample(q)
    np.testing.assert_array_equal(valid, [False, True, True, False, False])
    assert np.isnan(xy[~valid]).all()
    target.verify_contract(())


def test_past_departure_need_not_still_be_visible_to_supervise_current_pass():
    observed = observed_slice(30., 80.)
    obstacle = box(y=4.3)
    old = build_behavior_target(observed, [obstacle])
    assert not old.moving_supervision
    target = build_behavior_window(observed, [obstacle], source_ego_s=35.)
    assert target.moving_supervision, target.reasons
    assert len(target.excursions) == 1
    assert target.excursions[0].start_s < observed.reference.s[0]
    assert not target.diagnostics["complete_event_inside_query_span"]
    assert not target.sample(np.array([target.excursions[0].start_s]))[-1].any()
    assert target.diagnostics["minimum_road_edge_clearance_m"] >= .20
    assert target.diagnostics["minimum_obstacle_clearance_surplus_m"] >= 0.


def test_future_recovery_outside_visible_support_is_masked_without_masking_pass():
    target = build_behavior_window(observed_slice(30., 46.), [box(y=4.3)], source_ego_s=35.)
    assert target.moving_supervision, target.reasons
    event = target.excursions[0]
    assert event.end_s > target.road.reference.s[-1]
    q = np.array([35., 40., event.end_s, event.end_s+5.])
    np.testing.assert_array_equal(target.sample(q)[-1], [True, True, False, False])
    assert target.complete_event_qualified is False


def test_phase_anchors_do_not_move_to_the_current_window_start():
    obstacle = box(y=4.3)
    whole = build_behavior_window(road(), [obstacle], source_ego_s=15.)
    later = build_behavior_window(observed_slice(30.), [obstacle], source_ego_s=35.)
    assert whole.moving_supervision and later.moving_supervision
    assert whole.excursions == later.excursions
    q = np.linspace(35., 65., 61)
    np.testing.assert_allclose(whole.sample(q)[0], later.sample(q)[0], atol=1e-10)


def test_source_station_rebase_keeps_world_phase_anchors_and_geometry():
    observed = observed_slice(30.)
    ref = observed.reference
    rebased = ObservedRoad(ObservableReference(ref.left_xyz, ref.right_xyz, ref.s-30.,
        ref.evidence_frames, ref.decision_time_s))
    original = build_behavior_window(observed, [box(y=4.3)], source_ego_s=35.)
    local = build_behavior_window(rebased, [box(y=4.3)], source_ego_s=5.)
    assert original.moving_supervision and local.moving_supervision
    for name in ("start_s", "plateau_start_s", "plateau_end_s", "end_s"):
        assert getattr(original.excursions[0], name)-30. == pytest.approx(getattr(local.excursions[0], name))
    np.testing.assert_allclose(original.sample(np.array([35., 40., 60.]))[0],
        local.sample(np.array([5., 10., 30.]))[0], atol=1e-10)


def test_actual_ego_requires_observed_support_for_its_rear_body():
    target = build_behavior_window(observed_slice(30.), source_ego_s=31.)
    assert not target.moving_supervision
    assert target.reasons == ("actual_ego_rear_body_not_supported_by_observed_road",)
    assert not target.sample(np.array([31., 40.]))[-1].any()


def test_too_short_future_horizon_masks_without_extrapolating():
    target = build_behavior_window(road(), source_ego_s=79.)
    assert not target.moving_supervision
    assert target.reasons == ("insufficient_observed_future_body_support",)
    assert not target.sample(np.array([79., 100.]))[-1].any()


@pytest.mark.parametrize("motion", ["unknown", "moving"])
def test_dynamic_or_unknown_actor_is_not_ignored_even_when_behind(motion):
    target = build_behavior_window(road(), [box(y=4.3, motion=motion)], source_ego_s=60.)
    assert not target.moving_supervision
    assert target.reasons == ("obstacle_motion_unknown_or_dynamic:parked",)


def test_endpoint_clipped_actor_is_not_silently_assumed_behind():
    target = build_behavior_window(observed_slice(45.), [box(y=4.3)], source_ego_s=50.)
    assert not target.moving_supervision
    assert target.reasons == ("obstacle_station_not_supported_without_endpoint_clipping:parked",)
    assert target.diagnostics["actors_automatically_omitted"] == []


def test_causally_observed_past_actor_still_bound_to_nominal_recovery_window():
    obstacle = box(y=4.3)
    target = build_behavior_window(observed_slice(30.), [obstacle], source_ego_s=60.)
    assert target.moving_supervision, target.reasons
    assert target.obstacle_ids == ("parked",)
    np.testing.assert_allclose(target.xy[:, 1], 3.)
    with pytest.raises(ValueError, match="inputs or qualified solution changed"):
        target.verify_contract(())


@pytest.mark.parametrize("timestamp,reason", [(1., "missing_or_future"), (-5., "stale")])
def test_causal_actor_evidence_is_required_for_every_window(timestamp, reason):
    obstacle = box(y=4.3, evidence=(EvidenceFrame(13, timestamp, "wide"),))
    target = build_behavior_window(road(), [obstacle], source_ego_s=35.)
    assert not target.moving_supervision
    assert target.reasons == ("obstacle_evidence_"+reason+":parked",)


def test_narrow_road_never_uses_opposing_half_for_partial_pass():
    target = build_behavior_window(road(width=7.), [box(y=2.6)], source_ego_s=35.)
    assert not target.moving_supervision
    assert not target.sample(np.array([35., 40.]))[-1].any()


def test_window_contract_binds_current_station_solution_support_and_flags():
    target = build_behavior_window(road(), [box(y=4.3)], source_ego_s=35.)
    assert not target.s.flags.writeable
    for changed in (replace(target, source_ego_s=34.), replace(target, excursions=()),
                    replace(target, s=target.s+.1), replace(target, moving_supervision=False)):
        with pytest.raises(ValueError, match="inputs or qualified solution changed"):
            changed.verify_contract()
    with pytest.raises(ValueError, match="provenance contract changed"):
        replace(target, complete_event_qualified=True).verify_contract()
    target.road.reference._right.c[3, 0, 1] += .1
    with pytest.raises(ValueError, match="inputs or qualified solution changed"):
        target.sample(np.array([40.]))


def test_rigid_transform_keeps_window_and_obstacle_clearance():
    observed, obstacle = road(), box(y=4.3)
    original = build_behavior_window(observed, [obstacle], source_ego_s=35.)
    a = .4
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    translation = np.array([200., -50.])
    ref = observed.reference
    left, right = ref.left_xyz.copy(), ref.right_xyz.copy()
    left[:, :2] = left[:, :2]@rotation.T+translation
    right[:, :2] = right[:, :2]@rotation.T+translation
    transformed = ObservedRoad(ObservableReference(left, right, ref.s, ref.evidence_frames, ref.decision_time_s))
    target = build_behavior_window(transformed,
        [replace(obstacle, polygon_xy=obstacle.polygon()@rotation.T+translation)], source_ego_s=35.)
    assert target.moving_supervision
    np.testing.assert_allclose(target.xy, original.xy@rotation.T+translation, atol=1e-8)


def test_window_cannot_claim_full_episode_qa_or_accept_privileged_geometry():
    from lane_behavior.quality import qualify_episode
    target = build_behavior_window(road(), source_ego_s=20.)
    with pytest.raises(TypeError, match="BehaviorTarget"):
        qualify_episode(timestamps_s=[], positions_xyz=[], yaw_rad=[], velocity_world=[],
            lateral_acceleration_mps2=[], target=target)
    source = road().reference
    with pytest.raises(TypeError, match="observed"):
        build_behavior_window(ObservedRoad(LaneReference(source.left_xyz, source.right_xyz, source.s)),
            source_ego_s=20.)
    with pytest.raises(ValueError, match="actual ego station"):
        build_behavior_window(road(), source_ego_s=100.)
    with pytest.raises(ValueError, match="actual ego station"):
        build_behavior_window(road(), source_ego_s=True)
