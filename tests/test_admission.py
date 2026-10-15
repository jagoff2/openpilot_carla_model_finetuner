from dataclasses import asdict
import gzip
import json

import numpy as np
import pytest

from lane_dataset.admission import (PHASES, _plan_support, _verify_pixel_claims,
                                   build_labels, select_phase_indices, verify_qualification, command_domain_errors,
                                   interpolation_evidence, validate_reference_offset)
from lane_dataset.admission import causal_plan_support, _load_command_provenance
from lane_dataset.admission import qualify_candidate, validate_capture_envelope
from lane_dataset.adapter import dense_to_plan, PLAN_TIMES
from lane_dataset.controller import SteeringCalibration
from lane_dataset.geometry import MAX_OBSERVED_KNOT_GAP_M
from lane_dataset.config import Config, sha256_file, write_json
from lane_dataset.preprocessing import StockPreprocessor, reference_module
from lane_dataset.visibility import EvidenceRef, FrameEvidence, ObservableSupport


def test_projection_allows_sensor_float_roundoff_but_rejects_pixel_displacement():
    from types import SimpleNamespace
    from lane_dataset.admission import _verify_projection
    points=np.array([[11.5,-1.75,0.],[11.5,1.75,0.]])
    pixels=np.array([[964.-2648.*.175,604.+2648.*.13],
                     [964.+2648.*.175,604.+2648.*.13]])
    manifest=dict(camera_sensors={'narrow_rgb':dict(mount=dict(x=1.5,y=0,z=1.3,roll=0,pitch=0,yaw=0),focal_px=2648.)},
                  preprocessing=dict(model_to_camera=[np.eye(3).tolist(),np.eye(3).tolist()]))
    telemetry=dict(rotation_world=np.zeros((1,3)),position_world=np.zeros((1,3)))
    evidence=SimpleNamespace(reference=SimpleNamespace(camera='narrow'),crop_visible=np.zeros(2,bool),
        source_visible=np.ones(2,bool),source_pixels=pixels+.03,model_pixels=pixels)
    _verify_projection(evidence,manifest,telemetry,0,points)
    evidence.source_pixels=pixels+1.
    with pytest.raises(ValueError,match='do not project'):
        _verify_projection(evidence,manifest,telemetry,0,points)


def support_fixture(n=21, missing=None):
    ids = tuple([f"L{i}" for i in range(n)] + [f"R{i}" for i in range(n)])
    supported = np.ones(2*n, bool)
    if missing is not None:
        supported[missing] = False
    refs = tuple((EvidenceRef(100, 5.0, "narrow"),) if valid else () for valid in supported)
    return ObservableSupport(ids, np.tile(np.arange(n), 2), np.tile(np.arange(n), 2),
        np.ones(2*n, bool), supported, supported.copy(), supported.copy(), refs,
        np.ones(2*n), 20, 20, 20, 10, 0, True, "current", ())


def test_selection_spreads_four_labels_across_each_reviewed_phase():
    indices = np.arange(100, 500, 10)
    progress = np.arange(701) * .1
    ranges = {name: [10+8*i, 18+8*i] for i, name in enumerate(PHASES)}
    selected, phases, errors = select_phase_indices(indices, progress, ranges)
    assert not errors and len(selected) == 20 and np.all(selected % 10 == 0)
    assert all(list(phases.values()).count(phase) == 4 for phase in PHASES)


def test_bad_far_interpolation_masks_future_without_rejecting_valid_near_action():
    from lane_dataset.admission import interpolation_supports_action
    proof=dict(original_road_arc_m=[0.,10.,20.,30.],interval_geometry_valid=[True,True,False],
        spline_evidence_frames=[])
    assert interpolation_supports_action(proof,2.,3.)
    assert interpolation_supports_action(proof,19.,20.)
    assert not interpolation_supports_action(proof,19.,20.01)
    assert not interpolation_supports_action(proof,-1.,1.)
    telemetry=dict(timestamp=np.arange(301)*.05,route_s=np.arange(301)*.25)
    mask,_=_plan_support(dict(arc_m=[0.,10.,20.,30.]),telemetry,0,support_fixture(4),proof)
    query=PLAN_TIMES*5.
    assert mask[query<=20.].all()
    assert not mask[query>20.].any()


def reference_samples():
    yaw = np.linspace(-1, 2, 100)
    rate = np.where(np.arange(100) % 20 < 10, .12, -.18)
    lateral = 1.5 * rate
    velocity = np.column_stack((5*np.cos(yaw)-lateral*np.sin(yaw),
                                5*np.sin(yaw)+lateral*np.cos(yaw), np.zeros(100)))
    rotation = np.column_stack((np.zeros(100), np.zeros(100), yaw))
    return rotation, velocity, rate


def test_reference_offset_is_measured_with_independent_holdout():
    rotation, velocity, rate = reference_samples()
    result = validate_reference_offset(rotation,velocity,rate,3.,1.5)
    assert result["qualified"]
    assert result["fitted_reference_offset_m"] == pytest.approx(1.5)
    assert result["heldout_lateral_velocity_rmse_mps"] < 1e-12
    assert result["training_active_samples"] == 70
    assert result["heldout_active_samples"] == 30


def test_reference_offset_rejects_stale_default_or_tampered_value():
    rotation, velocity, rate = reference_samples()
    for stored in (0., 1.3, -1.5):
        result = validate_reference_offset(rotation,velocity,rate,3.,stored)
        assert not result["qualified"]
        assert "calibration_refit_mismatch:reference_offset_m" in result["errors"]


def test_reference_offset_holdout_cannot_change_training_estimate():
    rotation, velocity, rate = reference_samples()
    yaw = rotation[70:,2]
    velocity[70:,0] -= .08*np.sin(yaw)
    velocity[70:,1] += .08*np.cos(yaw)
    result = validate_reference_offset(rotation,velocity,rate,3.,1.5)
    assert result["fitted_reference_offset_m"] == pytest.approx(1.5)
    assert result["heldout_lateral_velocity_rmse_mps"] == pytest.approx(.08)
    assert "calibration_reference_offset_heldout_velocity_error" in result["errors"]


def test_reference_offset_requires_excitation_and_physical_range():
    rotation, velocity, rate = reference_samples()
    with pytest.raises(ValueError,match="active training and heldout"):
        validate_reference_offset(rotation,velocity,np.zeros(100),3.,1.5)
    result = validate_reference_offset(rotation,velocity,rate,1.,1.5)
    assert "calibration_reference_offset_outside_physical_range" in result["errors"]


@pytest.mark.parametrize("speed,curvature,horizon,reason", [
    (4.8, .01, .15, "command_speed_outside_measured_calibration_range"),
    (5.2, .01, .15, "command_speed_outside_measured_calibration_range"),
    (5.0, .3, .15, "command_outside_measured_steering_range"),
    (5.0, .01, .35, "recorded_lateral_action_horizon_mismatch"),
])
def test_command_requires_actual_calibration_domain_and_horizon(speed, curvature, horizon, reason):
    calibration = SteeringCalibration(3,1.08,1.22,.1,.03,4.95,5.05,.012,True,1000,
        validation_peak_yaw_rate_error_rps=.06,steering_fraction_min=-.25,
        steering_fraction_max=.25,steering_cubic_rad=-1.2,response_rate_rps=3.7)
    assert reason in command_domain_errors(calibration,speed,curvature,horizon)
    assert command_domain_errors(calibration,5.,.01,.15) == []


def test_absent_phase_semantics_remains_diagnostic():
    selected, phases, errors = select_phase_indices(np.arange(100, 500, 10), np.arange(700))
    assert len(selected) == 20 and errors == ["maneuver_phase_semantics_unreviewed"]
    assert set(phases.values()) == {"progress_only_unreviewed_phase"}


def test_plan_mask_does_not_reappear_after_unobserved_road_gap():
    geometry = {"arc_m": list(range(21))}
    telemetry = {"timestamp": np.arange(301)*.05, "route_s": np.arange(301)*.05}
    full, refs = _plan_support(geometry, telemetry, 0, support_fixture())
    assert full.all() and all(refs)
    mask, references = _plan_support(geometry, telemetry, 0, support_fixture(missing=5))
    first = np.flatnonzero(~mask)[0]
    assert mask[:first].all() and not mask[first:].any()
    assert all(not item for item in references[first:])
    # Altering unavailable later progress cannot restore a deterministic target.
    changed = {**telemetry, "route_s": telemetry["route_s"].copy()}
    changed["route_s"][200:] = 19
    perturbed, _ = _plan_support(geometry, changed, 0, support_fixture(missing=5))
    np.testing.assert_array_equal(mask, perturbed)


def causal_fixture():
    times = 5+np.arange(301)*.05
    frames = [dict(index=i,frame_id=100+i,timestamp=float(t)) for i,t in enumerate(times)]
    telemetry = dict(timestamp=times,command_accepted=np.ones(len(times),bool))
    provenance = [dict(**frame,command_accepted=True,source_indices=[0,1,2],
        evidence_frames=[dict(frame_id=frame["frame_id"],timestamp_s=frame["timestamp"],camera="narrow")])
        for frame in frames]
    return frames, telemetry, provenance


def test_legacy_capture_masks_future_and_interpolation_derived_initial_values():
    frames, telemetry, _ = causal_fixture()
    mask, report = causal_plan_support(frames,telemetry,0,np.ones(33,bool),support_fixture(),None)
    assert mask[0,0:6].all() and mask[0,9:12].all()
    assert not mask[0,6:9].any() and not mask[0,12:15].any() and not mask[1:].any()
    assert report["reason"] == "dense_command_provenance_missing"


def test_future_command_can_use_new_frame_of_already_known_geometry():
    frames, telemetry, provenance = causal_fixture()
    mask, report = causal_plan_support(frames,telemetry,0,np.ones(33,bool),support_fixture(),provenance)
    assert mask.all() and report["command_ledger_present"]
    assert report["supported_state_horizon_s"] == 10


def test_independent_boundary_command_causality_tracks_actual_side_knots():
    frames,telemetry,provenance=causal_fixture();support=support_fixture()
    support.supported[:]=False;support.supported[[0,2,22,24]]=True
    for record in provenance:record['source_point_indices']=[0,2,22,24]
    mask,report=causal_plan_support(frames,telemetry,0,np.ones(33,bool),support,provenance)
    assert mask.all()
    provenance[20]['source_point_indices']=[0,2,22,25]
    mask,report=causal_plan_support(frames,telemetry,0,np.ones(33,bool),support,provenance)
    assert report['first_unsupported_transition_index']==20
    assert report['reason']=='future_command_introduces_unseen_knots'
    assert not mask[PLAN_TIMES>1].any()


def test_hidden_future_control_perturbation_cannot_change_supervised_plan():
    frames, telemetry, provenance = causal_fixture()
    support = support_fixture(missing=10)
    barrier = 20
    provenance[barrier]["source_indices"] = [0,1,10]  # Newly visible road affects this command.
    masks, report = causal_plan_support(frames,telemetry,0,np.ones(33,bool),support,provenance)
    assert report["first_unsupported_transition_index"] == barrier
    assert report["reason"] == "future_command_introduces_unseen_knots"
    assert masks[:10].any() and not masks[PLAN_TIMES > 1].any()
    time = telemetry["timestamp"]
    position = np.column_stack((time-time[0],np.zeros(len(time)),np.zeros(len(time))))
    velocity = np.tile([1.,0,0],(len(time),1))
    rotation = np.zeros((len(time),3))
    base, valid = dense_to_plan(time,position,velocity,rotation,np.array([0]),masks[None])
    perturbed_position, perturbed_velocity, perturbed_rotation = position.copy(), velocity.copy(), rotation.copy()
    perturbed_position[barrier+1:,1] += np.linspace(.1,20,len(time)-barrier-1)
    perturbed_velocity[barrier+1:,1] += 4
    perturbed_rotation[barrier+1:,2] += .6
    changed, valid_changed = dense_to_plan(time,perturbed_position,perturbed_velocity,perturbed_rotation,
                                         np.array([0]),masks[None])
    np.testing.assert_array_equal(valid,valid_changed)
    np.testing.assert_array_equal(base[valid],changed[valid])
    assert np.any(base[~valid] != changed[~valid])


def test_interpolated_target_requires_next_endpoint_transition_even_near_origin():
    frames,telemetry,provenance = causal_fixture()
    provenance[0]["command_accepted"] = False
    mask, _ = causal_plan_support(frames,telemetry,0,np.ones(33,bool),support_fixture(),provenance)
    assert PLAN_TIMES[1] < .05
    assert mask[0,:6].all() and not mask[1].any()
    assert not mask[0,6:9].any() and not mask[0,12:15].any()


def test_dense_command_evidence_cannot_reference_a_future_frame():
    frames, telemetry, provenance = causal_fixture()
    provenance[0]["evidence_frames"][0] = dict(frame_id=101,timestamp_s=5.05,camera="narrow")
    mask, report = causal_plan_support(frames,telemetry,0,np.ones(33,bool),support_fixture(),provenance)
    assert not mask[1:].any()
    assert report["reason"] == "future_command_evidence_not_causal_or_aligned"


def test_exact_future_state_does_not_admit_outgoing_interpolation_derivatives():
    frames, telemetry, provenance = causal_fixture()
    provenance[50]["command_accepted"] = False
    mask, _ = causal_plan_support(frames,telemetry,0,np.ones(33,bool),support_fixture(),provenance)
    assert PLAN_TIMES[16] == 2.5
    assert mask[16,:6].all() and mask[16,9:12].all()
    assert not mask[16,6:9].any() and not mask[16,12:15].any()
    assert not mask[17:].any()


def test_dense_provenance_must_be_complete_aligned_and_hash_bound(tmp_path):
    frames,telemetry,provenance = causal_fixture()
    filename = "command-provenance.jsonl.gz"
    path = tmp_path/filename
    with gzip.open(path,"wt",encoding="utf-8") as stream:
        stream.write("\n".join(json.dumps(record) for record in provenance))
    manifest = dict(command_provenance_file=filename,files={filename:sha256_file(path)})
    assert _load_command_provenance(tmp_path,manifest,frames,telemetry) == provenance
    manifest["files"][filename] = "0"*64
    with pytest.raises(ValueError,match="hash"):
        _load_command_provenance(tmp_path,manifest,frames,telemetry)
    provenance[-1]["timestamp"] += .05
    with gzip.open(path,"wt",encoding="utf-8") as stream:
        stream.write("\n".join(json.dumps(record) for record in provenance))
    manifest["files"][filename] = sha256_file(path)
    with pytest.raises(ValueError,match="aligned"):
        _load_command_provenance(tmp_path,manifest,frames,telemetry)


def dashed_fixture():
    center = np.column_stack((np.arange(21), np.zeros(21), np.zeros(21)))
    geometry = dict(arc_m=list(range(21)),left_xyz=(center+[0,-2,0]).tolist(),right_xyz=(center+[0,2,0]).tolist())
    support = support_fixture()
    support.supported[:] = False
    support.supported[[0,4,8,12,16,20,21,25,29,33,37,41]] = True
    return geometry, support


def test_dashed_marking_interpolation_is_distinct_from_pixel_visibility():
    geometry, support = dashed_fixture()
    before = support.supported.copy()
    corridor = interpolation_evidence(geometry, support, [0,0,0])
    assert corridor["source_indices"] == [0,4,8,12,16,20]
    assert all(corridor["interval_geometry_valid"])
    np.testing.assert_array_equal(support.supported, before)
    telemetry = dict(timestamp=np.arange(301)*.05, route_s=np.arange(301)*.05)
    mask, references = _plan_support(geometry,telemetry,0,support,corridor)
    assert mask.all() and all(references)
    assert not support.supported[1]  # Interpolated support did not claim paint exists here.


def test_native_dashed_paint_gap_uses_shared_bound_and_retains_geometry_gate():
    n = 31
    center = np.column_stack((np.arange(n),np.zeros(n),np.zeros(n)))
    geometry = dict(arc_m=list(range(n)),left_xyz=(center+[0,-2,0]).tolist(),right_xyz=(center+[0,2,0]).tolist())
    support = support_fixture(n=n)
    knots = np.array([0,1,2,3,10,11,12,13,20,21,22,23])
    support.supported[:] = False
    support.supported[knots] = True
    support.supported[knots+n] = True
    corridor = interpolation_evidence(geometry,support,[0,0,0])
    assert corridor["maximum_marking_gap_m"] == MAX_OBSERVED_KNOT_GAP_M == 8.0
    assert corridor["source_indices"] == knots.tolist()
    assert all(corridor["interval_geometry_valid"])
    legacy = interpolation_evidence(geometry,support,[0,0,0],maximum_gap_m=5.)
    assert legacy["source_indices"] == [0,1,2,3]
    assert legacy["maximum_marking_gap_m"] != corridor["maximum_marking_gap_m"]
    for key in ("left_xyz","right_xyz"):
        geometry[key][6][1] += .1
    changed = interpolation_evidence(geometry,support,[0,0,0])
    assert not all(changed["interval_geometry_valid"])


def test_hidden_bend_between_observed_paint_knots_fails_geometric_validation():
    geometry, support = dashed_fixture()
    for key in ("left_xyz", "right_xyz"):
        geometry[key][2][1] += .2
    corridor = interpolation_evidence(geometry,support,[0,0,0])
    assert not corridor["interval_geometry_valid"][0]
    telemetry = dict(timestamp=np.arange(301)*.05, route_s=np.arange(301)*.05)
    mask, references = _plan_support(geometry,telemetry,0,support,corridor)
    assert not mask.any() and not any(references)


def test_interpolated_plan_never_extrapolates_before_first_observed_knot():
    geometry, support = dashed_fixture()
    support.supported[[0,21]] = False
    corridor = interpolation_evidence(geometry,support,[0,0,0])
    telemetry = dict(timestamp=np.arange(301)*.05, route_s=np.arange(301)*.05)
    mask, _ = _plan_support(geometry,telemetry,0,support,corridor)
    assert not mask.any()


def test_packed_pixels_must_support_ledger_marking_claim():
    packed = np.full((6, 128, 256), 16, np.uint8)
    reference = EvidenceRef(100, 5, "narrow")
    evidence = FrameEvidence(reference, ("L0",), np.array([1.]), np.ones(1, bool),
        np.ones(1, bool), np.array([[900., 600.]]), np.array([[100., 100.]]),
        np.array([100.]), np.array([100.]), np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="contrast"):
        _verify_pixel_claims(evidence, packed, np.zeros(3), np.array([[5, -2, 0]]))
    # One exact packed Y00 pixel supplies the claimed 100-luma contrast.
    packed[0, 50, 50] = 116
    _verify_pixel_claims(evidence, packed, np.zeros(3), np.array([[5, -2, 0]]))


def minimal_shard(root, compressed=True, frame_count=4):
    shard = root / "quarantine" / "development-fixture"
    shard.mkdir(parents=True)
    frames = [dict(index=i, frame_id=100+i, timestamp=5+i*.05, desire=[1,0,0,0,0,0,0,0],
                   action_t=[.1,.35], traffic_convention=[1,0]) for i in range(frame_count)]
    (shard / "frames.jsonl").write_text("\n".join(json.dumps(value) for value in frames))
    position = np.column_stack((np.arange(frame_count)*.05+1, np.zeros(frame_count), np.zeros(frame_count)))
    np.savez(shard / "telemetry.npz", frame_id=np.arange(100,100+frame_count), timestamp=5+np.arange(frame_count)*.05,
        position_world=position, velocity_world=np.tile([1.,0,0], (frame_count,1)), rotation_world=np.zeros((frame_count,3)),
        acceleration_world=np.zeros((frame_count,3)), speed=np.full(frame_count,1.), command_curvature=np.zeros(frame_count),
        command_accepted=np.ones(frame_count,bool), route_s=position[:,0], steer=np.zeros(frame_count))
    center = np.column_stack((np.arange(21), np.zeros(21), np.zeros(21)))
    write_json(shard / "geometry.json", dict(arc_m=list(range(21)), left_xyz=(center+[0,-2,0]).tolist(),
        right_xyz=(center+[0,2,0]).tolist(), section=dict(direction="straight")))
    fixture = dict(measured_m=[5.,6.,8.], camera_points=[[3,0,5],[4,0,6],[5,0,8]], depth_mode="forward")
    write_json(shard / "depth-fixture.json", {key: fixture for key in ("narrow","wide")})
    evidence_file = "evidence.jsonl.gz" if compressed else "evidence.jsonl"
    candidates_file = "candidates.json.gz" if compressed else "candidates.json"
    for name, value in ((evidence_file, ""), (candidates_file, "[]")):
        if compressed:
            with gzip.open(shard / name, "wt", encoding="utf-8") as stream:
                stream.write(value)
        else:
            (shard / name).write_text(value)
    archives = {}
    archive_module = reference_module("frame_archive_v2")
    for camera in ("narrow", "wide"):
        path = shard / (camera+".npz")
        with archive_module.FrameArchiveWriter(path, dtype=np.uint8, frame_shape=(6,128,256), expected_count=frame_count) as writer:
            for i in range(frame_count):
                packed = np.full((6,128,256), 16, np.uint8)
                packed[0,50,50] = 116
                writer.write(i, packed)
            writer.finalize(complete=True)
        archives[camera] = dict(path=path.name, sha256=sha256_file(path))
    manifest = dict(schema_version=1, status="development", episode_id="development-fixture", split="development",
        geometry_group_id="fixture", family="straight_recovery", frames_file="frames.jsonl", archives=archives,
        candidates_file=candidates_file, evidence_file=evidence_file, preprocessing=StockPreprocessor().contract(),
        config_sha256=Config().digest(), vehicle_dimensions=dict(length=4,width=1.8), collision_events=[],
        files={path.name:sha256_file(path) for path in shard.iterdir() if path.is_file()})
    write_json(shard / "manifest.json", manifest)
    return shard


@pytest.mark.parametrize("compressed", [False, True])
def test_no_candidates_produces_diagnostic_artifact_without_publishing(tmp_path, compressed):
    shard = minimal_shard(tmp_path, compressed)
    report = build_labels(shard, dataset_root=tmp_path)
    assert report["label_count"] == 0 and report["lateral_label_count"] == 0
    assert "supported_action_label_quota:0/20" in report["build_errors"]
    with np.load(shard / "labels.npz") as labels:
        assert labels["plan_targets"].shape == (0,33,15)
        assert labels["lateral_mask"].shape == (0,)
    manifest = json.loads((shard / "manifest.json").read_text())
    assert manifest["status"] == "development" and manifest["accepted_label_count"] == 0
    assert manifest["label_records"] == []


@pytest.mark.parametrize("three_dimensional", [False, True])
def test_admission_binds_3d_body_dimensions_and_keeps_legacy_capture_diagnostic(tmp_path, three_dimensional):
    shard = minimal_shard(tmp_path)
    manifest = json.loads((shard/"manifest.json").read_text())
    if three_dimensional:
        manifest["vehicle_dimensions"].update(height=1.5,bbox_center=[.1,0,.75])
    write_json(shard/"manifest.json",manifest)
    report = qualify_candidate(shard,dataset_root=tmp_path)
    assert not report["passed"]
    assert ("full_3d_vehicle_bounding_box_missing_or_invalid" in report["errors"]) != three_dimensional
    assert report["physical"]["metrics"]["body_footprint_mode"] == (
        "transformed_3d_bounding_box" if three_dimensional else "planar_fixture")


def test_tampered_capture_inputs_are_rejected_before_label_construction(tmp_path):
    shard = minimal_shard(tmp_path)
    with (shard / "telemetry.npz").open("ab") as stream: stream.write(b"tamper")
    with pytest.raises(ValueError, match="capture_artifact_hash"):
        build_labels(shard, dataset_root=tmp_path)


def test_boolean_gate_document_does_not_authorize_publication(tmp_path):
    write_json(tmp_path / "config.json", dict(frozen=True,sha256=Config().digest()))
    write_json(tmp_path / "qualification.json", dict(schema_version=1,config_sha256=Config().digest(),
        gates={key: True for key in ("controller","preprocessing","depth","worker_benchmark","split_overlap",
                                    "rendered_markings","runtime_action","hidden_road_perturbation")}))
    report = verify_qualification(tmp_path, dict(config_sha256=Config().digest()))
    assert not report["passed"]
    assert any("source_binding" in error for error in report["errors"])
    assert any("path and sha256" in error for error in report["errors"])


def test_end_to_end_retains_partial_plan_with_causal_evidence_and_masks_action(tmp_path):
    """Analytical pixels exercise admission plumbing, without claiming CARLA QA."""
    shard = minimal_shard(tmp_path, frame_count=301)
    n = 21
    ids = [f"L{i}" for i in range(n)] + [f"R{i}" for i in range(n)]
    visible = np.ones(2*n, bool)
    visible[[12, 12+n]] = False
    with gzip.open(shard / "evidence.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index in range(301):
            for camera in ("narrow", "wide"):
                doc = dict(reference=dict(frame_id=100+index,timestamp_s=5+index*.05,camera=camera),
                    point_ids=ids,arc_m=np.tile(np.arange(n),2).tolist(),source_visible=visible.tolist(),
                    crop_visible=visible.tolist(),source_pixels=np.tile([900.,600.],(2*n,1)).tolist(),
                    model_pixels=np.tile([100.,100.],(2*n,1)).tolist(),
                    source_contrast=np.full(2*n,100.).tolist(),crop_contrast=np.full(2*n,100.).tolist(),
                    source_depth_residual_m=np.zeros(2*n).tolist(),crop_depth_residual_m=np.zeros(2*n).tolist())
                stream.write(json.dumps(doc)+"\n")
    candidates = [dict(index=100,command_valid=True,support=dict(supported=visible.tolist()))]
    with gzip.open(shard / "candidates.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(candidates, stream)
    manifest = json.loads((shard / "manifest.json").read_text())
    for name in ("evidence.jsonl.gz", "candidates.json.gz"):
        manifest["files"][name] = sha256_file(shard/name)
    write_json(shard / "manifest.json", manifest)
    result = build_labels(shard, dataset_root=tmp_path)
    assert result["label_count"] == 1 and result["lateral_label_count"] == 0
    with np.load(shard / "labels.npz") as labels:
        mask = labels["plan_mask"][0,:,0]
        assert mask.any() and not mask.all()
        first = np.flatnonzero(~mask)[0]
        assert not mask[first:].any()
        assert not labels["lateral_mask"].any()
    record = json.loads((shard / "manifest.json").read_text())["label_records"][0]
    assert record["accepted"] is False and record["action_evidence_frames"] == []
    assert all(ref["timestamp_s"] <= record["timestamp_s"] for segment in record["plan_segment_evidence_frames"] for ref in segment)


def test_qualification_seal_detects_bootstrap_edits_and_new_operational_modules(tmp_path, monkeypatch):
    import lane_dataset.admission as admission
    project = tmp_path/"source"
    package = project/"lane_dataset"
    package.mkdir(parents=True)
    for name in ("bootstrap.py","config.py","scenario_planner.py"):
        (package/name).write_text("VALUE = 1\n")
    monkeypatch.setattr(admission,"PROJECT",project)
    root = tmp_path/"dataset"
    root.mkdir()
    config_hash = Config().digest()
    write_json(root/"config.json",dict(frozen=True,sha256=config_hash))
    write_json(root/"qualification.json",dict(schema_version=1,config_sha256=config_hash,
        sources={str(path.relative_to(project)):sha256_file(path) for path in package.glob("*.py")}))
    manifest = dict(config_sha256=config_hash)
    # Other evidence is intentionally absent; inspect only the independent seal.
    initial = verify_qualification(root,manifest)
    assert not any("source_" in error for error in initial["errors"])
    (package/"bootstrap.py").write_text("VALUE = 2\n")
    changed = verify_qualification(root,manifest)
    assert "qualification_source_binding_mismatch:bootstrap.py" in changed["errors"]
    (package/"traffic.py").write_text("VALUE = 1\n")
    added = verify_qualification(root,manifest)
    assert "qualification_source_inventory_mismatch" in added["errors"]
    assert "qualification_source_binding_mismatch:traffic.py" in added["errors"]


def test_qualification_seal_rejects_stale_removed_module_binding(tmp_path,monkeypatch):
    import lane_dataset.admission as admission
    project = tmp_path/"source"
    package = project/"lane_dataset"
    package.mkdir(parents=True)
    source = package/"config.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(admission,"PROJECT",project)
    root = tmp_path/"dataset"
    root.mkdir()
    config_hash = Config().digest()
    write_json(root/"config.json",dict(frozen=True,sha256=config_hash))
    write_json(root/"qualification.json",dict(schema_version=1,config_sha256=config_hash,
        sources={"lane_dataset/config.py":sha256_file(source),"lane_dataset/removed.py":"0"*64}))
    result = verify_qualification(root,dict(config_sha256=config_hash))
    assert "qualification_source_inventory_mismatch" in result["errors"]


def envelope_fixture(root,traffic_count=0):
    from lane_dataset.bootstrap import check_recording_seam
    raw = dict(frame_id=np.array([9,10]),timestamp=np.array([.45,.5]),
        rotation_world=np.zeros((2,3)),acceleration_world=np.tile([0.,.1,0.],(2,1)))
    telemetry = dict(frame_id=np.array([11,12,13]),timestamp=np.array([.55,.6,.65]),
        rotation_world=np.zeros((3,3)),acceleration_world=np.tile([0.,.2,0.],(3,1)))
    np.savez(root/"bootstrap-telemetry.npz",**raw)
    last = {key:value[-1] for key,value in raw.items()}
    first = {key:value[0] for key,value in telemetry.items()}
    manifest = dict(bootstrap=dict(ready=True,frames=2,last_frame_id=10,last_timestamp=.5,
        recording_seam=check_recording_seam(last,first)),traffic_count=traffic_count,
        files={"bootstrap-telemetry.npz":sha256_file(root/"bootstrap-telemetry.npz")})
    if traffic_count:
        report = dict(requested_count=traffic_count,spawned_count=traffic_count,
            spawn_records=[dict(actor_id=100+i) for i in range(traffic_count)],physics=True,
            hybrid_physics=False,synchronous=True,automatic_lane_change=False,passed=True,
            collisions=[],cleanup_errors=[],reasons=[],validated_frames=5,
            minimum_corridor_clearance_m=.4,minimum_body_clearance_m=.5)
        write_json(root/"traffic-report.json",report)
        manifest["traffic"] = report
        manifest["files"]["traffic-report.json"] = sha256_file(root/"traffic-report.json")
    return manifest,telemetry,raw


def test_capture_error_cannot_be_overridden_by_other_passing_evidence(tmp_path):
    manifest,telemetry,_ = envelope_fixture(tmp_path)
    assert validate_capture_envelope(tmp_path,manifest,telemetry)["passed"]
    manifest["error"] = {}
    result = validate_capture_envelope(tmp_path,manifest,telemetry)
    assert not result["passed"] and "capture_error_recorded" in result["errors"]


def test_seam_is_recomputed_from_bound_raw_samples_not_pass_boolean(tmp_path):
    manifest,telemetry,raw = envelope_fixture(tmp_path)
    raw["acceleration_world"][-1,1] = -.2  # Actual seam jerk is now 8 m/s^3.
    np.savez(tmp_path/"bootstrap-telemetry.npz",**raw)
    manifest["files"]["bootstrap-telemetry.npz"] = sha256_file(tmp_path/"bootstrap-telemetry.npz")
    result = validate_capture_envelope(tmp_path,manifest,telemetry)
    assert manifest["bootstrap"]["recording_seam"]["passed"]
    assert "bootstrap_recording_seam_lateral_jerk_above_limit" in result["errors"]
    assert "bootstrap_recording_seam_measurement_mismatch" in result["errors"]
    assert result["measurements"]["recording_seam"]["lateral_jerk_mps3"] == pytest.approx(8.)


@pytest.mark.parametrize("mutation", ["missing_hash","stale_hash","missing_seam","missing_frame","frame_count"])
def test_bootstrap_recording_seam_requires_bound_complete_evidence(tmp_path,mutation):
    manifest,telemetry,raw = envelope_fixture(tmp_path)
    if mutation == "missing_hash": manifest["files"].clear()
    elif mutation == "stale_hash": manifest["files"]["bootstrap-telemetry.npz"] = "0"*64
    elif mutation == "missing_seam": del manifest["bootstrap"]["recording_seam"]
    elif mutation == "missing_frame": telemetry["frame_id"][0] += 1
    else: manifest["bootstrap"]["frames"] += 1
    result = validate_capture_envelope(tmp_path,manifest,telemetry)
    assert not result["passed"]
    assert any("bootstrap" in error for error in result["errors"])


@pytest.mark.parametrize("field,value", [
    ("requested_count",1),("spawned_count",1),("physics",False),("hybrid_physics",True),
    ("synchronous",False),("automatic_lane_change",True),("passed",False),
    ("collisions",[dict(frame_id=12)]),("cleanup_errors",["actor destroy failed"]),
    ("validated_frames",4),("minimum_corridor_clearance_m",-.01),("minimum_body_clearance_m",.14),
])
def test_traffic_admission_requires_full_physical_safety_and_frame_coverage(tmp_path,field,value):
    manifest,telemetry,_ = envelope_fixture(tmp_path,traffic_count=2)
    assert validate_capture_envelope(tmp_path,manifest,telemetry)["passed"]
    manifest["traffic"][field] = value
    write_json(tmp_path/"traffic-report.json",manifest["traffic"])
    manifest["files"]["traffic-report.json"] = sha256_file(tmp_path/"traffic-report.json")
    result = validate_capture_envelope(tmp_path,manifest,telemetry)
    assert not result["passed"]
    assert any("traffic" in error for error in result["errors"])


def test_traffic_artifact_cannot_be_unbound_or_hidden_by_zero_count(tmp_path):
    manifest,telemetry,_ = envelope_fixture(tmp_path,traffic_count=2)
    del manifest["files"]["traffic-report.json"]
    assert not validate_capture_envelope(tmp_path,manifest,telemetry)["passed"]
    manifest["traffic_count"] = 0
    result = validate_capture_envelope(tmp_path,manifest,telemetry)
    assert "traffic_evidence_present_but_declared_count_zero" in result["errors"]
