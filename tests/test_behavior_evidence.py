from dataclasses import replace

import numpy as np
import pytest

from lane_behavior.evidence import (ActorState, EvidenceConfig, FrameStamp, MODALITIES,
    Observation, build_static_proof, decode_carla_instance_bgra, observation_proof,
    pack_model_rgb, verify_static_proof, visible_surface_points)


FOOTPRINT = np.array([[-2., -3.], [2., -3.], [2., 3.], [-2., 3.]])


def scene(category="parked_vehicle", *, frame=10, timestamp=0., instance_id=513):
    stamp = FrameStamp(frame, timestamp, "wide")
    rgb = np.full((256, 512, 3), 45, np.uint8)
    rgb[80:160, 240:280] = [180, 80, 50]
    semantic = np.zeros(rgb.shape[:2], np.uint8)
    semantic[80:160, 240:280] = 14 if category == "parked_vehicle" else 12
    instances = np.zeros(rgb.shape[:2], np.uint16)
    instances[80:160, 240:280] = instance_id
    raw = np.zeros((*rgb.shape[:2], 4), np.uint8)
    raw[..., 2] = semantic
    raw[..., 1] = (instances % 256).astype(np.uint8)
    raw[..., 0] = (instances // 256).astype(np.uint8)
    raw[..., 3] = 255
    warp = np.eye(3, dtype=np.float32)
    return Observation(stamp, {k:stamp for k in MODALITIES}, rgb,
        np.full(rgb.shape[:2], 10., np.float32), semantic, instances,
        pack_model_rgb(rgb, warp), np.array([[200., 0., 256.], [0., 200., 128.], [0., 0., 1.]]),
        np.eye(4), warp, "forward", "a"*64, True,
        instance_bgra=raw, scene_actor_ids=(1, 2, instance_id))


def history(category="parked_vehicle"):
    observations = [scene(category, frame=10+5*i, timestamp=.25*i) for i in range(3)]
    states = [ActorState(o.stamp, 513, (0., 0., 10.), 0., (0., 0., 0.), 0.) for o in observations]
    return observations, states


def kwargs(observations, category="parked_vehicle"):
    return dict(actor_id=513, instance_id=513, category=category,
        conservative_footprint_world_xy=FOOTPRINT, decision=observations[-1].stamp)


def individual(observation, **overrides):
    args = dict(actor_id=513, instance_id=513, category="parked_vehicle",
        conservative_footprint_world_xy=FOOTPRINT)
    args.update(overrides)
    return observation_proof(observation, **args)


@pytest.mark.parametrize("category", ["parked_vehicle", "pedestrian"])
def test_static_visual_presence_is_replayable_but_does_not_claim_hidden_extent_or_safe_pass(category):
    observations, states = history(category)
    args = kwargs(observations, category)
    proof = build_static_proof(observations, states, **args)
    assert proof["supported"] and proof["outcome"] == "static_obstacle_observed"
    assert not proof["entire_footprint_observed"]
    assert not proof["safe_to_pass"] and not proof["future_motion_assumed"]
    assert proof["geometry_source"] == "visible_metric_depth_surface"
    assert proof["footprint_source"] == "conservative_simulator_qa_envelope"
    assert verify_static_proof(proof, observations, states, **args) == proof
    assert all(r["source_pixels"] == 3200 and r["crop_pixels"] == 3200 for r in proof["observations"])
    assert len(proof["observations"][-1]["visible_surface_world_xyz"]) == 512


def test_actual_packed_bytes_are_checked_for_luma_and_chroma():
    observation = scene()
    for channel in (0, 4, 5):
        broken = observation.packed.copy()
        broken[channel, 50, 120] ^= 1
        with pytest.raises(ValueError, match="actual packed"):
            individual(replace(observation, packed=broken))


def test_native_rgb_crop_replay_matches_existing_collector_on_nontrivial_warp():
    from lane_dataset.preprocessing import image_to_nv12, warp_and_pack_nv12
    rng = np.random.default_rng(551)
    rgb = rng.integers(0, 256, (320, 640, 3), dtype=np.uint8)
    warp = np.array([[1.10001, .013, 1.5], [-.015, 1.0802, 3.5], [.00001, -.00003, 1.]], np.float32)
    source, layout = image_to_nv12(rgb, "RGB")
    assert np.array_equal(pack_model_rgb(rgb, warp), warp_and_pack_nv12(source, layout, warp))


@pytest.mark.parametrize("modality", MODALITIES)
def test_cross_frame_or_cross_camera_arrays_cannot_be_claimed_as_same_observation(modality):
    observation = scene()
    stamps = dict(observation.modality_stamps)
    stamps[modality] = replace(observation.stamp, frame_id=11)
    with pytest.raises(ValueError, match="exact same frame"):
        individual(replace(observation, modality_stamps=stamps))
    stamps[modality] = replace(observation.stamp, camera="narrow")
    with pytest.raises(ValueError, match="exact same frame"):
        individual(replace(observation, modality_stamps=stamps))


def test_source_visible_but_outside_actual_crop_cannot_support_obstacle():
    observation = scene()
    warp = np.eye(3, dtype=np.float32)
    warp[0, 2] = 300
    changed = replace(observation, model_to_source=warp, packed=pack_model_rgb(observation.rgb, warp))
    proof = individual(changed)
    assert proof["source_pixels"] == 3200 and proof["crop_pixels"] == 0
    assert not proof["supported"] and "crop_instance_unresolved" in proof["reasons"]


def test_clamped_edge_pixels_do_not_invent_model_crop_support():
    observation = scene()
    rgb = np.full_like(observation.rgb, 45)
    rgb[:, -1] = [180, 80, 50]
    semantic = np.zeros_like(observation.semantic_ids); semantic[:, -1] = 14
    instance = np.zeros_like(observation.instance_ids); instance[:, -1] = 513
    raw = np.zeros_like(observation.instance_bgra)
    raw[..., 2] = semantic; raw[..., 1] = instance % 256; raw[..., 0] = instance // 256
    warp = np.eye(3, dtype=np.float32); warp[0, 2] = 600
    changed = replace(observation, rgb=rgb, semantic_ids=semantic, instance_ids=instance,
        instance_bgra=raw, model_to_source=warp, packed=pack_model_rgb(rgb, warp))
    assert individual(changed)["crop_pixels"] == 0


def test_semantic_only_wrong_class_and_raw_instance_disagreement_reject():
    observation = scene()
    with pytest.raises(ValueError, match="semantic-only"):
        individual(replace(observation, instance_ids=None))
    with pytest.raises(ValueError, match="semantic category"):
        individual(observation, category="pedestrian")
    raw = observation.instance_bgra.copy(); raw[100, 250, 0] = 0
    with pytest.raises(ValueError, match="raw instance sensor"):
        individual(replace(observation, instance_bgra=raw))


def test_instance_byte_order_and_namespace_are_explicit():
    raw = np.array([[[2, 1, 14, 255]]], np.uint8)
    semantic, ids = decode_carla_instance_bgra(raw)
    assert semantic.item() == 14 and ids.item() == 513
    with pytest.raises(ValueError, match="16-bit instance"):
        individual(scene(), actor_id=66049)
    with pytest.raises(ValueError, match="16-bit scene"):
        individual(replace(scene(), scene_actor_ids=(513, 66049)))
    with pytest.raises(ValueError, match="16-bit scene"):
        individual(replace(scene(), scene_actor_ids=(1, 2)))


def test_semantic_mask_without_rgb_contrast_is_not_visual_evidence():
    observation = scene()
    rgb = np.full_like(observation.rgb, 45)
    proof = individual(replace(observation, rgb=rgb, packed=pack_model_rgb(rgb, observation.model_to_source)))
    assert not proof["supported"] and "crop_rgb_contrast_unresolved" in proof["reasons"]


def test_invalid_or_unqualified_depth_rejects_and_hidden_footprint_cannot_replace_it():
    observation = scene()
    with pytest.raises(ValueError, match="depth fixture"):
        individual(replace(observation, depth_qualified=False))
    depth = observation.depth_m.copy(); depth[80:160, 240:280] = np.nan
    proof = individual(replace(observation, depth_m=depth))
    assert not proof["supported"] and proof["visible_surface_world_xyz"] == []
    assert "crop_instance_depth_unresolved" in proof["reasons"]


def test_surface_depth_is_checked_against_conservative_qa_envelope():
    observation = scene()
    depth = observation.depth_m.copy(); depth[80:160, 240:280] = 50.
    proof = individual(replace(observation, depth_m=depth))
    assert not proof["supported"]
    assert "visible_surface_outside_conservative_qa_footprint" in proof["reasons"]
    with pytest.raises(ValueError, match="convex"):
        individual(observation, conservative_footprint_world_xy=FOOTPRINT[[0, 2, 1, 3]])


def test_backprojection_uses_selected_depth_convention_and_world_transform():
    observation = scene()
    pixels = np.array([[256, 128], [280, 128]])
    forward = visible_surface_points(observation, pixels)
    assert np.allclose(forward[:, 2], 10.)
    radial = visible_surface_points(replace(observation, depth_mode="radial"), pixels)
    assert np.allclose(np.linalg.norm(radial, axis=1), 10.)
    transform = np.eye(4); transform[0, 3] = -3.
    moved = visible_surface_points(replace(observation, camera_from_world=transform), pixels)
    assert np.allclose(moved-forward, [3., 0., 0.])


@pytest.mark.parametrize("change, reason", [
    ({"velocity_world_mps": (.1, 0., 0.)}, "actor_moving_yield"),
    ({"position_world_m": (.1, 0., 10.)}, "actor_displacement_yield"),
    ({"yaw_rad": .1}, "actor_rotation_yield"),
    ({"actor_id": 514}, "actor_stationarity_identity_mismatch"),
    ({"source": "assumed_static"}, "actor_stationarity_identity_mismatch"),
])
def test_moving_or_unbound_actor_history_yields(change, reason):
    observations, states = history()
    states[-1] = replace(states[-1], **change)
    proof = build_static_proof(observations, states, **kwargs(observations))
    assert proof["outcome"] == "yield" and reason in proof["reasons"]


def test_missing_stale_and_future_history_yields():
    observations, states = history()
    proof = build_static_proof(observations[-1:], states[-1:], **kwargs(observations))
    assert "stationary_history_insufficient" in proof["reasons"]
    proof = build_static_proof(observations, states[:1], **kwargs(observations))
    assert "aligned_actor_state_history_missing" in proof["reasons"]
    args = kwargs(observations); args["decision"] = observations[0].stamp
    proof = build_static_proof(observations, states, **args)
    assert "future_mixed_camera_or_unordered_history" in proof["reasons"]
    observations[1] = replace(observations[1], stamp=replace(observations[1].stamp, timestamp_s=.01))
    proof = build_static_proof(observations, states, **kwargs(observations))
    assert "stationary_history_gap" in proof["reasons"]


def test_carla_float32_tick_roundoff_does_not_create_history_gap():
    observations = [scene(frame=10+5*i, timestamp=5.644034918397665+i*0.2500000037252903)
        for i in range(3)]
    states = [ActorState(o.stamp, 513, (0., 0., 10.), 0., (0., 0., 0.), 0.) for o in observations]
    proof = build_static_proof(observations, states, **kwargs(observations))
    assert proof['supported'], proof['reasons']


def test_changed_proof_or_payload_fails_replay():
    observations, states = history()
    args = kwargs(observations)
    proof = build_static_proof(observations, states, **args)
    changed = dict(proof, entire_footprint_observed=True)
    with pytest.raises(ValueError, match="does not match"):
        verify_static_proof(changed, observations, states, **args)
    observations[0] = replace(observations[0], depth_m=observations[0].depth_m+.02)
    with pytest.raises(ValueError, match="does not match"):
        verify_static_proof(proof, observations, states, **args)


def test_config_rejects_nonfinite_and_noninteger_thresholds():
    with pytest.raises(ValueError):
        EvidenceConfig(maximum_static_speed_mps=float("nan"))
    with pytest.raises(ValueError):
        EvidenceConfig(minimum_static_frames=2)
    with pytest.raises(ValueError):
        EvidenceConfig(minimum_crop_pixels=1.5)
