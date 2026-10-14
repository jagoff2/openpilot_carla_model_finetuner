from dataclasses import replace

import numpy as np
import pytest

from lane_dataset.preprocessing import (_project_plane_nearest, image_to_nv12,
                                        split_nv12, warp_and_pack_nv12, StockPreprocessor)
from lane_dataset.visibility import (CameraObservation, EvidenceHistory, EvidenceRef,
    FrameEvidence, probe_lane_evidence, decode_carla_depth_bgra, validate_depth_geometry)


def observation(x=100, *, frame_id=0, timestamp_s=0., camera="narrow", occluded=False):
    rgb = np.full((320, 640, 3), 40, np.uint8)
    rgb[70:130, x-1:x+2] = 240
    depth = np.full((320, 640), 10., np.float32)
    if occluded:
        depth[95:106, x-5:x+6] = 3.
    return CameraObservation(frame_id, timestamp_s, camera, rgb, depth, np.eye(4),
        np.diag([100., 100., 1.]), np.eye(3, dtype=np.float32), depth_validated=True)


def evidence(frame_id, time, camera, visible, source=None, ids=("near", "curve")):
    visible = np.asarray(visible, bool)
    source = visible.copy() if source is None else np.asarray(source, bool)
    n = len(visible)
    return FrameEvidence(EvidenceRef(frame_id, time, camera), ids, np.arange(n),
        source, visible, np.zeros((n, 2)), np.zeros((n, 2)), np.full(n, 50.),
        np.full(n, 50.), np.zeros(n), np.zeros(n))


def get_support(history, frame=10, time=.5, required=(True, True), ids=("near", "curve")):
    return history.support(frame, time, ids, np.array(required), np.array([0., 2.]),
                           ["near"], distance_m=np.array([5., 20.]))


def test_frustum_without_rendered_marking_is_not_evidence():
    obs = observation()
    blank = replace(obs, rgb=np.full_like(obs.rgb, 40))
    result = probe_lane_evidence(blank, [[10., 10., 10.]], ["p"], [10.])
    assert not result.crop_visible.any()
    assert not result.source_visible.any()


def test_depth_occlusion_and_rendered_marking_are_both_required():
    points = [[10., 10., 10.]]
    clear = probe_lane_evidence(observation(), points, ["p"], [10.])
    occluded = probe_lane_evidence(observation(occluded=True), points, ["p"], [10.])
    assert clear.crop_visible.tolist() == [True]
    assert clear.source_visible.tolist() == [True]
    assert not occluded.crop_visible.any()
    assert occluded.crop_depth_residual_m[0] == pytest.approx(7.)


def test_source_evidence_outside_crop_never_becomes_model_support():
    result = probe_lane_evidence(observation(x=550), [[55., 10., 10.]], ["p"], [30.])
    assert result.source_visible.tolist() == [True]
    assert result.crop_visible.tolist() == [False]


def test_semantic_marking_mask_can_only_remove_support():
    obs = observation()
    result = probe_lane_evidence(replace(obs, marking_mask=np.zeros((320,640),bool)),
                                 [[10.,10.,10.]], ["p"], [10.])
    assert not result.source_visible.any() and not result.crop_visible.any()


def test_unvalidated_depth_is_rejected():
    with pytest.raises(ValueError, match="fixture-qualified"):
        probe_lane_evidence(replace(observation(), depth_validated=False), [[10.,10.,10.]], ["p"], [10.])


def test_reused_canonical_luma_matches_conversion_and_validates_shape(monkeypatch):
    obs=observation()
    baseline=probe_lane_evidence(obs,[[10.,10.,10.]],["p"],[10.])
    nv12,layout=image_to_nv12(obs.rgb,"RGB")
    luma,_,_=split_nv12(nv12,layout)
    def conversion_must_not_repeat(*args,**kwargs):
        raise AssertionError("reused source luma triggered another full RGB conversion")
    monkeypatch.setattr("lane_dataset.preprocessing.image_to_nv12",conversion_must_not_repeat)
    reused=probe_lane_evidence(replace(obs,luma=luma),[[10.,10.,10.]],["p"],[10.])
    np.testing.assert_array_equal(reused.crop_visible,baseline.crop_visible)
    np.testing.assert_array_equal(reused.crop_contrast,baseline.crop_contrast)
    with pytest.raises(ValueError,match="precomputed luma"):
        probe_lane_evidence(replace(obs,luma=np.zeros((256,512),np.uint8)),[[10.,10.,10.]],["p"],[10.])


@pytest.mark.parametrize("camera_index", [0, 1])
def test_luma_visibility_sampler_is_exact_stock_packed_luma(camera_index):
    pipeline = StockPreprocessor(calibration_rpy=(.012, -.008, .015))
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (1208,1928,3), dtype=np.uint8)
    nv12, layout = image_to_nv12(rgb, "RGB")
    y, _, _ = split_nv12(nv12, layout)
    warp = pipeline.warps[camera_index]
    packed = warp_and_pack_nv12(nv12, layout, warp)
    expected = np.empty((256,512), np.uint8)
    expected[0::2,0::2] = packed[0]
    expected[1::2,0::2] = packed[1]
    expected[0::2,1::2] = packed[2]
    expected[1::2,1::2] = packed[3]
    np.testing.assert_array_equal(_project_plane_nearest(y,warp,(512,256)), expected)


def test_crop_history_loss_requires_measured_duration_and_source_preview():
    history = EvidenceHistory()
    for frame, time in [(0, 0.), (1,.05), (2,.10)]:
        for camera in ("narrow","wide"):
            history.admit(evidence(frame,time,camera,[True,True]))
    for camera in ("narrow","wide"):
        history.admit(evidence(10,.5,camera,[True,False],[True,True]))
    support = get_support(history)
    assert support.category == "history_crop_loss"
    assert support.action_valid and support.crop_loss_qualified
    np.testing.assert_allclose(support.visibility_duration_s, [.1,.1])
    assert support.preview_distance_m == 5.
    assert support.historical_preview_distance_m == 20.
    assert support.crop_preview_loss_m == 15.
    assert support.supported_horizon_s == 2.
    assert all(ref.timestamp_s <= .5 for row in support.evidence_frames for ref in row)


def test_occluded_current_source_cannot_fill_crop_loss_quota():
    history = EvidenceHistory()
    for frame, time in [(0,0.),(1,.05),(2,.1)]:
        for camera in ("narrow","wide"):
            history.admit(evidence(frame,time,camera,[True,True]))
    for camera in ("narrow","wide"):
        history.admit(evidence(10,.5,camera,[True,False],[True,False]))
    assert get_support(history).category == "history_supported"


def test_fully_ambiguous_geometry_is_diagnostic_and_action_mask_is_independent():
    history = EvidenceHistory()
    for camera in ("narrow","wide"):
        history.admit(evidence(10,.5,camera,[True,False],[True,True]))
    support = get_support(history)
    assert support.category == "diagnostic_ambiguous"
    assert support.action_valid
    assert support.supported.tolist() == [True, False]
    optional_future = get_support(history, required=(True,False))
    assert optional_future.category == "current"
    assert optional_future.supported_horizon_s == 0.


def test_future_evidence_and_geometry_ids_cannot_retroactively_qualify_decision():
    history = EvidenceHistory()
    for camera in ("narrow","wide"):
        history.admit(evidence(10,.5,camera,[True,False]))
    for camera in ("narrow","wide"):
        history.admit(evidence(11,.55,camera,[True,True]))
    assert get_support(history).supported.tolist() == [True,False]
    assert get_support(history).category == "diagnostic_ambiguous"


def test_expired_history_cannot_support_action():
    history = EvidenceHistory()
    for camera in ("narrow","wide"):
        history.admit(evidence(0,0.,camera,[True,True]))
    for camera in ("narrow","wide"):
        history.admit(evidence(100,5.,camera,[False,False]))
    support = get_support(history,frame=100,time=5.)
    assert not support.action_valid
    assert not support.supported.any()


def test_missing_camera_and_duplicate_frame_rejected():
    history = EvidenceHistory()
    frame = evidence(10,.5,"narrow",[True,True])
    history.admit(frame)
    with pytest.raises(ValueError, match="duplicate"):
        history.admit(frame)
    with pytest.raises(ValueError, match="both stock"):
        get_support(history)


def test_visibility_duration_does_not_bridge_unobserved_time():
    history = EvidenceHistory()
    for frame, time in [(0,0.),(8,.4)]:
        for camera in ("narrow","wide"):
            history.admit(evidence(frame,time,camera,[True,True]))
    for camera in ("narrow","wide"):
        history.admit(evidence(10,.5,camera,[True,False],[True,True]))
    support = get_support(history)
    assert support.visibility_duration_s[1] == 0
    assert not support.crop_loss_qualified


def test_known_geometry_depth_convention_qualification_and_decode():
    points = np.array([[0.,0.,10.],[5.,0.,10.],[0.,6.,10.]])
    report = validate_depth_geometry(np.array([10.,10.,10.]), points)
    assert report["qualified"] and report["depth_mode"] == "forward"
    assert validate_depth_geometry(np.linalg.norm(points,axis=1),points)["depth_mode"] == "radial"
    assert not validate_depth_geometry(np.array([10.,10.,10.]),np.tile([0.,0.,10.],(3,1)))["qualified"]
    bgra = np.array([[[0,0,0,255],[255,255,255,255]]],np.uint8)
    np.testing.assert_allclose(decode_carla_depth_bgra(bgra),[[0.,1000.]])


def test_hidden_road_perturbation_does_not_change_admitted_spline():
    from lane_dataset.geometry import LaneReference
    s = np.arange(8., dtype=float)
    left = np.column_stack((s,np.full(8,-1.75),np.zeros(8)))
    right = np.column_stack((s,np.full(8,1.75),np.zeros(8)))
    mask = np.array([True]*5+[False]*3)
    refs = [(EvidenceRef(0,0.,"narrow"),)]*5+[()]*3
    baseline = LaneReference(left,right,s).admit(mask,refs,0.)
    changed_left, changed_right = left.copy(), right.copy()
    changed_left[5:,1] += [1.,2.,4.]
    changed_right[5:,1] += [1.,2.,4.]
    perturbed = LaneReference(changed_left,changed_right,s).admit(mask,refs,0.)
    for one,two in zip(baseline.sample(np.linspace(0,4,17)),perturbed.sample(np.linspace(0,4,17))):
        np.testing.assert_array_equal(one,two)
