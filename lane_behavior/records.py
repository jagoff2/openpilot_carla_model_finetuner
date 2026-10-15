"""Portable, replayed behavior labels kept separate from episode admission.

These records are additional lateral targets. They do not overwrite the original
lane geometry, grant acceptance, or become additional camera-model inputs.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np

from lane_dataset.geometry import EvidenceFrame, LaneReference, ObservableReference
from .target import BehaviorConfig, EgoFootprint, ObservedObstacle, ObservedRoad, build_behavior_target
from .window import build_behavior_window
from .passing import ObservedPassingSpace,transformed_observed_space

SCHEMA = "unmarked_lateral_behavior_record_v1"


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def build_record(road, obstacles=(), ego=None, *, config=None, label_frame_id,
                 label_timestamp_s, evidence_bindings, ego_position_world, ego_yaw_rad,
                 target_mode="complete_event",passing_space=None):
    """Serialize a deterministic target from causal observed geometry.

    Evidence bindings must point to immutable proof/artifact digests. This is a
    geometry record, not an assertion that those external artifacts passed QA.
    The full episode publisher must independently replay them before admission.
    """
    if not isinstance(road, ObservedRoad):
        raise TypeError("observed whole-road corridor required")
    ref = road.reference
    if (not isinstance(label_frame_id, int) or isinstance(label_frame_id, bool)
            or label_frame_id < 0 or not np.isfinite(label_timestamp_s)
            or abs(ref.decision_time_s-label_timestamp_s) > 1e-6):
        raise ValueError("label decision time must match observed geometry")
    refs = [r for row in ref.evidence_frames for r in row]
    refs.extend(r for o in obstacles for r in o.evidence_frames)
    if passing_space is not None:
        if not isinstance(passing_space,ObservedPassingSpace):raise TypeError('observed passing-space proof required')
        refs.extend(passing_space.evidence_frames)
    if any(not isinstance(r.frame_id, int) or r.frame_id > label_frame_id for r in refs):
        raise ValueError("future or nonnumeric evidence frame")
    if (not isinstance(evidence_bindings, dict) or not evidence_bindings
            or any(not isinstance(name, str) or not name or not isinstance(sha, str)
                or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                for name, sha in evidence_bindings.items())):
        raise ValueError("named SHA256 evidence bindings required")
    if passing_space is not None:
        if (evidence_bindings.get('controlled_scene')!=passing_space.controlled_scene_sha256
                or evidence_bindings.get('rendered_free_space')!=passing_space.rendered_free_space_sha256):
            raise ValueError('passing permission lacks independently named scene and rendered-free-space bindings')
    position = np.asarray(ego_position_world, float)
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(ego_yaw_rad):
        raise ValueError("finite bound ego pose required")
    config, ego = config or BehaviorConfig(), ego or EgoFootprint()
    if target_mode not in ("complete_event", "observed_future_window", "observed_future_geometry"):
        raise ValueError("unknown behavior target mode")
    source_ego_s = None
    if target_mode == "observed_future_window":
        nominal = LaneReference((ref.left_xyz+ref.right_xyz)*.5,ref.right_xyz,ref.s)
        source_ego_s,_ = nominal.project(position[:2])
        target = build_behavior_window(road,obstacles,ego,config=config,source_ego_s=source_ego_s,passing_space=passing_space)
    elif target_mode == "observed_future_geometry":
        target = build_behavior_window(road,obstacles,ego,config=config,passing_space=passing_space)
    else:
        target = build_behavior_target(road, obstacles, ego, config=config,passing_space=passing_space)
    record = dict(schema=SCHEMA, label_frame_id=label_frame_id, label_timestamp_s=float(label_timestamp_s),
        coordinate_frame="carla_world_xy_positive_y_right", ego_position_world=position.tolist(),
        ego_yaw_rad=float(ego_yaw_rad), ego_footprint=asdict(ego), config=asdict(config),
        road=dict(left_role=road.left_role, right_role=road.right_role, s=ref.s.tolist(),
            left_xyz=ref.left_xyz.tolist(), right_xyz=ref.right_xyz.tolist(),
            evidence_frames=[[asdict(r) for r in row] for row in ref.evidence_frames]),
        obstacles=[dict(identity=o.identity, polygon_xy=o.polygon().tolist(), kind=o.kind,
            motion=o.motion, evidence_frames=[asdict(r) for r in o.evidence_frames]) for o in obstacles],
        evidence_bindings=dict(evidence_bindings), target_version=target.version,
        target_mode=target_mode,source_ego_s=source_ego_s,
        moving_supervision=target.moving_supervision, reasons=list(target.reasons),
        excursions=[asdict(e) for e in target.excursions], target_diagnostics=target.diagnostics,
        artifact_scope="geometry_only_requires_independent_visual_and_physical_episode_admission",
        training_ready=False)
    if passing_space is not None:record['passing_space']=passing_space.document()
    # Canonicalize dataclass tuple fields before computing/replaying hashes.
    record = json.loads(json.dumps(record, allow_nan=False))
    record["record_sha256"] = digest(record)
    return record


def replay_record(record, *, verified_evidence_bindings):
    """Reject changed geometry/metadata; require independently supplied bindings."""
    if record.get("schema") != SCHEMA:
        raise ValueError("unsupported behavior record")
    document = {k: v for k, v in record.items() if k != "record_sha256"}
    if record.get("record_sha256") != digest(document):
        raise ValueError("behavior record digest mismatch")
    if record["evidence_bindings"] != verified_evidence_bindings:
        raise ValueError("behavior proof/artifact binding mismatch")
    data = record["road"]
    evidence = [[EvidenceFrame(**r) for r in row] for row in data["evidence_frames"]]
    road = ObservedRoad(ObservableReference(data["left_xyz"], data["right_xyz"], data["s"],
        evidence, record["label_timestamp_s"]), data["left_role"], data["right_role"])
    obstacles = tuple(ObservedObstacle(o["identity"], o["polygon_xy"], o["kind"],
        tuple(EvidenceFrame(**r) for r in o["evidence_frames"]), o["motion"]) for o in record["obstacles"])
    ego = EgoFootprint(**record["ego_footprint"])
    config = BehaviorConfig(**record["config"])
    passing_space=None if record.get('passing_space') is None else ObservedPassingSpace.from_document(record['passing_space'])
    replay = build_record(road, obstacles, ego, config=config,
        label_frame_id=record["label_frame_id"], label_timestamp_s=record["label_timestamp_s"],
        evidence_bindings=verified_evidence_bindings, ego_position_world=record["ego_position_world"],
        ego_yaw_rad=record["ego_yaw_rad"],target_mode=record.get("target_mode","complete_event"),passing_space=passing_space)
    if digest(replay) != digest(record):
        raise ValueError("recorded behavior disagrees with deterministic reconstruction")
    return road, obstacles, ego, config


def training_corridor(record, *, verified_evidence_bindings):
    """Rebuild the target in current-ego FRD; caller separately admits episode."""
    from .loss import BehaviorCorridor
    road, obstacles, ego, config = replay_record(record,
        verified_evidence_bindings=verified_evidence_bindings)
    origin = np.asarray(record["ego_position_world"])
    yaw = record["ego_yaw_rad"]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    def local3(points):
        result = np.asarray(points).copy()-origin
        result[:, :2] = result[:, :2]@rotation
        return result
    ref = road.reference
    local_road = ObservedRoad(ObservableReference(local3(ref.left_xyz), local3(ref.right_xyz),
        ref.s, ref.evidence_frames, ref.decision_time_s), road.left_role, road.right_role)
    local_obstacles = tuple(ObservedObstacle(o.identity, (o.polygon()-origin[:2])@rotation,
        o.kind, o.evidence_frames, o.motion) for o in obstacles)
    space=None if record.get('passing_space') is None else ObservedPassingSpace.from_document(record['passing_space'])
    local_space=transformed_observed_space(space,ref,local_road.reference)
    if record.get("target_mode") in ("observed_future_window", "observed_future_geometry"):
        target = build_behavior_window(local_road,local_obstacles,ego,config=config,source_ego_s=record["source_ego_s"],passing_space=local_space)
        if not target.moving_supervision:
            return None
        return BehaviorCorridor(target,record["source_ego_s"],local_obstacles)
    target = build_behavior_target(local_road, local_obstacles, ego, config=config,passing_space=local_space)
    if not target.moving_supervision:
        return None
    # Correspondence follows nominal right-side road progress, never nearest
    # student path or a potentially different avoidance branch.
    nominal = target.nominal_xy
    segment = np.diff(nominal, axis=0)
    fraction = np.clip(-(nominal[:-1]*segment).sum(1)/np.maximum((segment*segment).sum(1), 1e-12), 0, 1)
    index = np.argmin(np.linalg.norm(nominal[:-1]+fraction[:, None]*segment, axis=1))
    ego_s = float(target.s[index]+fraction[index]*(target.s[index+1]-target.s[index]))
    return BehaviorCorridor(target, ego_s, local_obstacles)
