from dataclasses import replace
import json

import numpy as np
import pytest

from lane_dataset.geometry import LaneReference, ObservableReference
from lane_behavior.planned import (PlannedObstacle, PlannedTarget, SOURCE_SCOPE,
    planned_scene_geometry)
from lane_behavior.target import (BehaviorTarget, EgoFootprint, build_behavior_target)
from test_behavior_target import box, road


def simulator_road(width=12., length=80.):
    source = road(width=width, length=length).reference
    return LaneReference(source.left_xyz, source.right_xyz, source.s)


def simulator_obstacle(**kwargs):
    source = box(**kwargs)
    return PlannedObstacle(source.identity, source.polygon(), source.kind, source.motion)


def test_plain_simulator_geometry_plans_teacher_without_any_observation_objects():
    reference = simulator_road()
    assert not isinstance(reference, ObservableReference)
    assert not hasattr(reference, "evidence_frames")
    target = planned_scene_geometry(reference, source_metadata={"map_sha256": "a"*64})
    assert isinstance(target, PlannedTarget) and not isinstance(target, BehaviorTarget)
    assert target.feasible and target.supervision_allowed is False
    assert target.source_scope == SOURCE_SCOPE
    for name in ("evidence_frames", "moving_supervision", "road"):
        assert not hasattr(target, name)
    assert np.allclose(target.xy[:, 1], 3.)
    document = target.document()
    assert not document["training_ready"] and not document["physical_qa_passed"]
    assert "evidence_frames" not in json.dumps(document, allow_nan=False)
    assert document["source"]["source_metadata"]["map_sha256"] == "a"*64


def test_planned_avoidance_retains_whole_body_clearance_and_complete_nominal_return():
    obstacle = simulator_obstacle(y=4.3)
    target = planned_scene_geometry(simulator_road(), [obstacle])
    assert target.feasible, target.reasons
    assert len(target.excursions) == 1
    event = target.excursions[0]
    assert event.offset_m < 0 and event.end_s+5. <= target.s[-1]
    points, direction, *_ = target.sample(np.array([event.start_s-1., event.plateau_start_s,
        event.plateau_end_s, event.end_s, event.end_s+5.]))
    assert np.allclose(points[[0, 3, 4], 1], 3.)
    assert np.allclose(direction[[3, 4]], [[1., 0.], [1., 0.]])
    assert target.diagnostics["minimum_road_edge_clearance_m"] >= .2
    assert target.diagnostics["minimum_obstacle_clearance_surplus_m"] >= 0.


def test_shared_geometry_matches_observed_solver_without_sharing_evidence_contract():
    observed_road = road()
    observed_obstacle = box(y=4.3)
    observed = build_behavior_target(observed_road, [observed_obstacle])
    planned = planned_scene_geometry(simulator_road(), [simulator_obstacle(y=4.3)])
    assert observed.moving_supervision and planned.feasible
    assert observed.excursions == planned.excursions
    assert np.array_equal(observed.xy, planned.xy)
    assert observed.contract_sha256 != planned.contract_sha256


def test_profile_readback_is_geometry_not_an_admitted_record():
    target = planned_scene_geometry(simulator_road(), [simulator_obstacle(y=4.3)])
    profile = target.profile()
    assert len(profile["position_xyz"]) == len(target.s)
    assert np.all(np.diff(profile["path_arc_m"]) > 0)
    assert np.isfinite(profile["curvature_1pm"]).all()
    assert np.allclose(np.asarray(profile["position_xyz"])[:, :2], target.xy)
    assert np.allclose(np.asarray(profile["virtual_divider_xy"])[:, 1], 0.)
    assert np.allclose(np.asarray(profile["right_road_edge_xy"])[:, 1], 6.)
    assert profile["scope"] == "planned_geometry_not_physical_trajectory_or_observed_label"


@pytest.mark.parametrize("motion", ["unknown", "moving"])
def test_unverified_static_actor_does_not_create_teacher_avoidance(motion):
    target = planned_scene_geometry(simulator_road(), [simulator_obstacle(motion=motion)])
    assert not target.feasible
    assert "planned_obstacle_motion_unknown_or_dynamic:parked" in target.reasons
    assert not target.sample(np.array([10., 30.]))[-1].any()
    assert target.document()["planned_profile"] is None


def test_narrow_road_yields_instead_of_entering_opposing_half():
    target = planned_scene_geometry(simulator_road(width=7.), [simulator_obstacle(y=2.6)])
    assert not target.feasible
    assert "no_planned_right_half_pass_with_complete_recovery" in target.reasons


def test_obstacle_near_route_end_requires_five_metres_after_complete_return():
    # At x=60 the old target can finish its recovery before the support ends,
    # but cannot retain another five metres of correctly offset nominal path.
    target = planned_scene_geometry(simulator_road(), [simulator_obstacle(x=60., y=4.3)])
    assert not target.feasible
    assert any(r.startswith("insufficient_planned_departure_return_and_recovery") for r in target.reasons)
    with pytest.raises(ValueError, match="five metres"):
        planned_scene_geometry(simulator_road(), minimum_retained_nominal_recovery_m=4.)


def test_multiple_nearby_actors_are_avoided_as_one_complete_event():
    target = planned_scene_geometry(simulator_road(length=120.),
        [simulator_obstacle(name="first", x=45., y=4.3), simulator_obstacle(name="second", x=56., y=4.3)])
    assert target.feasible, target.reasons
    assert len(target.excursions) == 1
    assert target.excursions[0].obstacle_ids == ("first", "second")
    assert target.excursions[0].end_s+5. <= target.s[-1]


def test_source_geometry_and_solver_result_mutation_invalidates_teacher_contract():
    reference = simulator_road()
    target = planned_scene_geometry(reference, [simulator_obstacle(y=4.3)])
    with pytest.raises(ValueError, match="changed after"):
        replace(target, excursions=()).sample(np.array([40.]))
    with pytest.raises(ValueError, match="provenance contract"):
        replace(target, supervision_allowed=True).verify_contract()
    reference._right.c[3, 0, 1] += .1
    with pytest.raises(ValueError, match="changed after"):
        target.profile()


def test_declared_simulator_envelope_cannot_be_replaced_after_qualification():
    obstacle = simulator_obstacle(y=4.3)
    target = planned_scene_geometry(simulator_road(), [obstacle])
    obstacle.polygon_xy[0, 1] += .1
    with pytest.raises(ValueError, match="changed after"):
        target.sample(np.array([40.]))


def test_rigid_transform_preserves_planned_behavior():
    reference, obstacle = simulator_road(), simulator_obstacle(y=4.3)
    original = planned_scene_geometry(reference, [obstacle])
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = np.array([300., -80.])
    left, right = reference.left_xyz.copy(), reference.right_xyz.copy()
    left[:, :2] = left[:, :2]@rotation.T+translation
    right[:, :2] = right[:, :2]@rotation.T+translation
    transformed = planned_scene_geometry(LaneReference(left, right, reference.s),
        [replace(obstacle, polygon_xy=obstacle.polygon_xy@rotation.T+translation)])
    assert transformed.feasible
    assert np.allclose(transformed.xy, original.xy@rotation.T+translation, atol=1e-8)


def test_full_body_dimensions_not_just_center_points_constrain_the_teacher():
    target = planned_scene_geometry(simulator_road(width=7.), ego=EgoFootprint(center_xy=(0., -1.)))
    assert not target.feasible
    assert "vehicle_footprint_leaves_right_half" in target.reasons


def test_teacher_target_cannot_enter_observed_target_or_admission_interfaces():
    from lane_behavior.control import build_control_reference
    from lane_behavior.quality import qualify_episode
    from lane_behavior.records import build_record
    target = planned_scene_geometry(simulator_road())
    with pytest.raises(TypeError, match="observed"):
        build_behavior_target(target)
    with pytest.raises(TypeError, match="BehaviorTarget"):
        build_control_reference(target)
    with pytest.raises(TypeError, match="BehaviorTarget"):
        qualify_episode(timestamps_s=[], positions_xyz=[], yaw_rad=[], velocity_world=[],
            lateral_acceleration_mps2=[], target=target)
    with pytest.raises(TypeError, match="observed"):
        build_record(target, label_frame_id=1, label_timestamp_s=0., evidence_bindings={},
            ego_position_world=[0., 0., 0.], ego_yaw_rad=0.)


def test_observed_obstacle_objects_cannot_silently_be_used_as_oracle_objects():
    with pytest.raises(TypeError, match="PlannedObstacle"):
        planned_scene_geometry(simulator_road(), [box()])


def test_teacher_target_cannot_enter_differentiable_observed_loss():
    pytest.importorskip("torch")
    from lane_behavior.loss import BehaviorCorridor
    target = planned_scene_geometry(simulator_road())
    with pytest.raises(TypeError, match="observed BehaviorTarget"):
        BehaviorCorridor(target, float(target.s[0]))


def curved_obstacle_scene():
    # Physical Tesla/Audi dimensions from the native static probe. Curvature
    # rotates the ego beside the Audi, beyond the lateral-width-only seed.
    s = np.linspace(0., 100., 501)
    angle = s/60.
    center = np.c_[60.*np.sin(angle), 60.*(np.cos(angle)-1.), s*0.]
    normal = np.c_[np.sin(angle), np.cos(angle), s*0.]
    reference = LaneReference(center-4.*normal, center+4.*normal, s)
    angle = 50./60.
    tangent, normal = np.array([np.cos(angle), -np.sin(angle)]), np.array([np.sin(angle), np.cos(angle)])
    origin = reference._right(50.)[:2]+normal*(.89434-.8)
    corners = np.array([[-1.85268, -.89434], [1.85268, -.89434],
                        [1.85268, .89434], [-1.85268, .89434]])
    polygon = origin+corners[:, :1]*tangent+corners[:, 1:]*normal
    obstacle = PlannedObstacle("parked", polygon, "parked_vehicle", "static")
    ego = EgoFootprint(4.79178, 2.16345, (.02921, 0.))
    return reference, obstacle, ego


def test_curved_pass_refines_lateral_seed_without_reducing_body_clearance(monkeypatch):
    import lane_behavior.planned as planned
    reference, obstacle, ego = curved_obstacle_scene()
    target = planned_scene_geometry(reference, [obstacle], ego)
    assert target.feasible, target.reasons
    assert target.diagnostics["attempted_alternatives"] == 2
    assert target.diagnostics["offset_refinement_step_m"] == .05
    assert target.diagnostics["minimum_road_edge_clearance_m"] >= .20
    assert target.diagnostics["minimum_obstacle_clearance_surplus_m"] >= 0.
    monkeypatch.setattr(planned, "_clearance_offsets",
        lambda low, high, *_: sorted((low-.03, high+.03), key=abs))
    old_search = planned_scene_geometry(reference, [obstacle], ego)
    assert not old_search.feasible
    assert "insufficient_obstacle_body_clearance:parked" in old_search.reasons


def test_curved_refinement_respects_explicit_search_budget():
    from lane_behavior.target import BehaviorConfig
    reference, obstacle, ego = curved_obstacle_scene()
    target = planned_scene_geometry(reference, [obstacle], ego,
        config=BehaviorConfig(maximum_alternatives=1))
    assert not target.feasible
    assert target.reasons == ("planned_avoidance_search_budget_exhausted",)
    assert target.diagnostics["attempted_alternatives"] == 1
