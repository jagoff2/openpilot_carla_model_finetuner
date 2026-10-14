"""Evidence-bound label construction and fail-closed shard admission.

Label construction never publishes a shard or supplies a review on a user's
behalf. The final admission report binds every input artifact; publication must
re-run it immediately before the same-volume atomic move.
"""
from __future__ import annotations

from dataclasses import fields
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .adapter import PLAN_TIMES, dense_to_plan, lateral_action_target
from .config import Config, PROJECT, sha256_file, write_json
from .controller import SteeringCalibration, fit_steering_calibration, fit_rate_limited_calibration
from .geometry import LaneReference, MAX_OBSERVED_KNOT_GAP_M
from .loader import validate_frame_metadata
from .preprocessing import DEPENDENCY_ROOT, StockPreprocessor, reference_module
from .postprocess import shard_postprocess_lock
from .quality import QualityConfig, qualify_episode
from .visibility import EvidenceHistory, EvidenceRef, FrameEvidence, validate_depth_geometry

PHASES = ("approach", "entry", "apex", "exit", "recovery")
GATES = ("controller", "preprocessing", "depth", "worker_benchmark", "split_overlap",
         "rendered_markings", "runtime_action", "hidden_road_perturbation")
CRITICAL_SOURCES = tuple(sorted((PROJECT / "lane_dataset").glob("*.py")))


def _json(path):
    path = Path(path)
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as stream:
        return json.load(stream)


def _member(root, name):
    root = Path(root).resolve()
    result = (root / name).resolve()
    if root not in result.parents:
        raise ValueError("artifact escapes shard directory")
    return result


def _binding(binding, base):
    if not isinstance(binding, dict) or not binding.get("path") or not binding.get("sha256"):
        raise ValueError("artifact needs path and sha256 binding")
    path = Path(binding["path"])
    path = path.resolve() if path.is_absolute() else (Path(base) / path).resolve()
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def _root_for(shard, dataset_root):
    return Path(dataset_root).resolve() if dataset_root is not None else Path(shard).resolve().parents[1]


def validate_reference_offset(rotation_world, velocity_world, yaw_rate, wheelbase_m, stored_offset_m):
    """Independently fit the actor-to-bicycle reference correction on training.

    The final chronological 30% is used only for validation. Legacy zero/default
    offsets cannot qualify a production corpus merely because steering fits.
    """
    rotation, velocity, rate = (np.asarray(value, float) for value in (rotation_world, velocity_world, yaw_rate))
    if (rotation.ndim != 2 or rotation.shape[1] != 3 or velocity.shape != rotation.shape
        or rate.shape != (len(rotation),) or len(rate) < 10
        or not all(np.isfinite(value).all() for value in (rotation, velocity, rate))
        or not np.isfinite([wheelbase_m, stored_offset_m]).all() or wheelbase_m <= 0):
        raise ValueError("reference offset requires finite aligned physical calibration samples")
    yaw = rotation[:, 2]
    lateral = -np.sin(yaw)*velocity[:, 0] + np.cos(yaw)*velocity[:, 1]
    cut = int(len(rate)*.7)
    active = np.abs(rate) > .05
    train = active & (np.arange(len(rate)) < cut)
    holdout = active & (np.arange(len(rate)) >= cut)
    if train.sum() < 2 or holdout.sum() < 2:
        raise ValueError("reference offset requires active training and heldout yaw samples")
    offset = float(np.sum(lateral[train]*rate[train])/np.sum(rate[train]**2))
    rmse = float(np.sqrt(np.mean((lateral[holdout]-offset*rate[holdout])**2)))
    errors = []
    if not np.isfinite(offset) or not 0 < offset < wheelbase_m:
        errors.append("calibration_reference_offset_outside_physical_range")
    if not np.isfinite(rmse) or rmse > .03:
        errors.append("calibration_reference_offset_heldout_velocity_error")
    if not np.isclose(stored_offset_m, offset, rtol=1e-5, atol=1e-7):
        errors.append("calibration_refit_mismatch:reference_offset_m")
    return dict(qualified=not errors, errors=errors, fitted_reference_offset_m=offset,
                heldout_lateral_velocity_rmse_mps=rmse, training_active_samples=int(train.sum()),
                heldout_active_samples=int(holdout.sum()), chronological_training_fraction=.7)


def _calibration_status(manifest, root):
    """Refit the actual maneuver samples, rather than trusting `qualified` alone."""
    errors = []
    try:
        path = _binding(manifest["calibration"], root)
        stored = SteeringCalibration(**_json(path))
        samples = path.with_name("samples.npz")
        start = None
        metadata_path = path.with_name("manifest.json")
        if not samples.is_file() and metadata_path.is_file():
            captures = _json(metadata_path).get("captures", [])
            matches = [capture for capture in captures if
                       np.isclose(capture.get("speed_min_mps", -1), stored.speed_min_mps) and
                       np.isclose(capture.get("speed_max_mps", -1), stored.speed_max_mps)]
            if len(matches) != 1:
                raise ValueError("cannot uniquely bind calibration to measured speed capture")
            samples = _member(path.parent, matches[0]["file"])
            start = int(matches[0]["warmup_frames"])
        with np.load(samples, allow_pickle=False) as data:
            start = len(data["timestamp"]) - 1 - stored.sample_count if start is None else start
            if start < 0 or len(data["timestamp"]) - 1 - start != stored.sample_count:
                raise ValueError("calibration sample_count does not identify the recorded fitting interval")
            fit = fit_rate_limited_calibration if stored.response_rate_rps < 99 or stored.steering_cubic_rad else fit_steering_calibration
            # Capture applies its command before ticking. Shift that command to
            # the previous measured state, as the steering model identification does.
            fitted = fit(data["timestamp"][start:-1], data["command_steer"][start+1:], data["speed"][start:-1],
                data["angular_velocity_world"][start:-1, 2], stored.wheelbase_m, stored.max_steering_rad)
            reference_validation = validate_reference_offset(data["rotation_world"][start:-1],
                data["velocity_world"][start:-1], data["angular_velocity_world"][start:-1, 2],
                stored.wheelbase_m, stored.reference_offset_m)
            errors.extend(reference_validation["errors"])
        for key in ("steering_gain_rad", "delay_s", "time_constant_s", "speed_min_mps", "speed_max_mps",
                    "validation_yaw_rate_rmse_rps", "validation_peak_yaw_rate_error_rps", "steering_cubic_rad", "response_rate_rps"):
            if not np.isclose(getattr(stored, key), getattr(fitted, key), rtol=1e-5, atol=1e-7):
                errors.append("calibration_refit_mismatch:" + key)
        if not fitted.qualified or not stored.qualified:
            errors.append("measured_steering_calibration_not_qualified")
        return stored, errors, {"fit": {"path": str(path), "sha256": sha256_file(path)},
                                "samples": {"path": str(samples), "sha256": sha256_file(samples)}}
    except (KeyError, ValueError, TypeError, OSError) as error:
        return None, ["calibration_evidence_invalid:" + str(error)], {}


def _depth_status(shard):
    errors = []
    try:
        report = _json(shard / "depth-fixture.json")
        for camera in ("narrow", "wide"):
            value = report[camera]
            recalculated = validate_depth_geometry(np.asarray(value["measured_m"]),
                                                  np.asarray(value["camera_points"]), tolerance_m=.15)
            if not recalculated["qualified"] or recalculated["depth_mode"] != value.get("depth_mode"):
                errors.append("depth_fixture_failed:" + camera)
    except (KeyError, ValueError, TypeError, OSError) as error:
        errors.append("depth_fixture_invalid:" + str(error))
    return errors


def rendered_boundary_status(shard, manifest, telemetry):
    from .rendered_boundaries import summarize_frames
    names=("rendered-boundary-report.json","rendered-boundary-samples.json.gz")
    for name in names:
        expected=manifest.get("files",{}).get(name)
        if not expected or sha256_file(shard/name)!=expected:
            raise ValueError("rendered boundary artifact missing or changed: "+name)
    report=_json(shard/names[0]);frames=_json(shard/names[1])
    if report.get("samples_sha256")!=sha256_file(shard/names[1]):
        raise ValueError("rendered boundary samples are not bound to their report")
    if [row["frame_id"] for row in frames]!=telemetry["frame_id"].tolist():
        raise ValueError("rendered boundary measurements do not cover every retained frame")
    replay=summarize_frames(frames)
    if any(report.get(key)!=value for key,value in replay.items()):
        raise ValueError("rendered boundary report differs from replayed stripe measurements")
    return replay


def command_domain_errors(calibration, speed, curvature, action_horizon_s):
    """Check actual measured speed/steering domains and recorded command horizon."""
    errors = []
    if not np.isfinite([speed, curvature, action_horizon_s]).all():
        return ["expert_command_or_horizon_nonfinite"]
    if not (speed > .3 and calibration.speed_min_mps <= speed <= calibration.speed_max_mps):
        errors.append("command_speed_outside_measured_calibration_range")
    if abs(action_horizon_s - (calibration.delay_s + .05)) > 1e-6:
        errors.append("recorded_lateral_action_horizon_mismatch")
    angle = np.arctan(calibration.wheelbase_m * curvature)
    try:
        fraction = calibration.steer_from_angle(angle)
    except ValueError:
        fraction = np.inf
    if not calibration.steering_fraction_min <= fraction <= calibration.steering_fraction_max:
        errors.append("command_outside_measured_steering_range")
    return errors


def _luma(packed):
    value = np.empty((256, 512), np.uint8)
    value[0::2, 0::2], value[1::2, 0::2] = packed[0], packed[1]
    value[0::2, 1::2], value[1::2, 1::2] = packed[2], packed[3]
    return value


def _verify_pixel_claims(evidence, packed, positions, boundary_points):
    """Recheck admitted marking pixels in authenticated model inputs.

    Depth residuals remain capture measurements, bound to their original ledger
    and independently qualified depth fixture. Full depth images are not stored.
    """
    image = _luma(packed)
    yy, xx = np.mgrid[-4:5, -4:5]
    flank = np.maximum(np.abs(xx), np.abs(yy)) >= 3
    for i in np.flatnonzero(evidence.crop_visible):
        pixel = evidence.model_pixels[i]
        if not np.isfinite(pixel).all():
            raise ValueError("admitted crop pixel is nonfinite")
        x, y = np.rint(pixel).astype(int)
        if x < 4 or y < 4 or x >= 508 or y >= 252:
            raise ValueError("admitted crop pixel outside valid marking patch")
        kind=1 if evidence.crop_boundary_kind is None else evidence.crop_boundary_kind[i]
        if kind==2:
            from .boundary_edges import edge_patch
            if evidence.crop_edge_road_bits is None or evidence.crop_edge_other_bits is None:
                raise ValueError('admitted edge lacks semantic patch proof')
            road=np.unpackbits(np.asarray(evidence.crop_edge_road_bits[i],np.uint8))[:81].reshape(9,9).astype(bool)
            other=np.unpackbits(np.asarray(evidence.crop_edge_other_bits[i],np.uint8))[:81].reshape(9,9).astype(bool)
            edge=edge_patch(image[y-4:y+5,x-4:x+5],road,other)
            if edge is None or edge[0]<12 or not np.isclose(edge[0],evidence.crop_contrast[i],atol=1e-4):
                raise ValueError('admitted edge contrast does not match packed frame')
        elif kind==3:
            from .boundary_color import paint_color_patch
            if evidence.crop_paint_mask_bits is None:
                raise ValueError('admitted color paint lacks semantic patch proof')
            mask=np.unpackbits(np.asarray(evidence.crop_paint_mask_bits[i],np.uint8))[:81].reshape(9,9).astype(bool)
            chroma=np.stack((packed[4],packed[5]),axis=-1).repeat(2,axis=0).repeat(2,axis=1)
            paint=paint_color_patch(image[y-4:y+5,x-4:x+5],chroma[y-4:y+5,x-4:x+5],mask)
            from .visibility import EvidenceThresholds
            if paint is None or paint[0]<EvidenceThresholds().minimum_chroma_contrast or not np.isclose(paint[0],evidence.crop_contrast[i],atol=1e-4):
                raise ValueError('admitted color paint contrast does not match packed frame')
        elif kind==1:
            peak = float(image[y-1:y+2, x-1:x+2].max())
            contrast = peak - float(np.percentile(image[y-4:y+5, x-4:x+5][flank].astype(np.float32), 60))
            if peak < 70 or contrast < 12 or not np.isclose(contrast, evidence.crop_contrast[i], atol=1e-4):
                raise ValueError("admitted marking contrast does not match packed frame")
        else:
            raise ValueError('admitted boundary has unknown detector kind')
        source = evidence.source_pixels[i]
        if not (np.isfinite(source).all() and 0 <= source[0] < 1928 and 0 <= source[1] < 1208):
            raise ValueError("crop claim originates outside source image")
        # Conservative upper bound accounts for camera translation and optical
        # forward/radial conventions without silently declaring depth pixels kept.
        maximum_expected = np.linalg.norm(boundary_points[i] - positions) + 3.0
        residual = evidence.crop_depth_residual_m[i]
        if not np.isfinite(residual) or residual < 0 or residual > .2 + .015 * maximum_expected:
            raise ValueError("admitted crop depth residual is invalid")
        if kind==2:
            if evidence.crop_edge_other_depth_residual_m is None:
                raise ValueError('admitted edge lacks exterior depth proof')
            residual=evidence.crop_edge_other_depth_residual_m[i]
            if not np.isfinite(residual) or residual<0 or residual>.2+.015*maximum_expected:
                raise ValueError('admitted edge exterior depth residual is invalid')


def _verify_projection(evidence, manifest, telemetry, index, boundary_points):
    """Bind ledger pixel coordinates to recorded ego pose and camera mounting."""
    from scipy.spatial.transform import Rotation
    sensor = manifest["camera_sensors"][evidence.reference.camera + "_rgb"]
    mount = sensor["mount"]
    reflect = np.diag([1., 1., -1.])
    vehicle = reflect @ Rotation.from_euler("xyz", telemetry["rotation_world"][index]).as_matrix() @ reflect
    mounting = reflect @ Rotation.from_euler("xyz", np.deg2rad([mount["roll"], mount["pitch"], mount["yaw"]])).as_matrix() @ reflect
    camera_rotation = vehicle @ mounting
    location = telemetry["position_world"][index] + vehicle @ np.array([mount["x"], mount["y"], mount["z"]])
    carla_camera = (boundary_points-location) @ camera_rotation
    optical = carla_camera[:, [1, 2, 0]] * [1, -1, 1]
    pixels = np.full((len(optical), 2), np.nan)
    np.divide(optical[:, :2], optical[:, 2:3], out=pixels, where=np.abs(optical[:, 2:3]) > 1e-12)
    pixels = pixels * float(sensor["focal_px"]) + [964., 604.]
    visible = evidence.crop_visible | evidence.source_visible
    # CARLA serializes actor pose and sensor inverse matrices independently as
    # float32. At Town04 world coordinates their roundoff differs by up to
    # 0.028 pixels on the measured bend. A twentieth of a pixel covers that
    # serialization error while still rejecting a displaced camera/ledger.
    if not np.allclose(pixels[visible], evidence.source_pixels[visible], rtol=1e-6, atol=.05):
        raise ValueError("ledger source pixels do not project recorded boundary geometry")
    camera_index = 0 if evidence.reference.camera == "narrow" else 1
    inverse_warp = np.linalg.inv(np.asarray(manifest["preprocessing"]["model_to_camera"][camera_index], np.float32))
    projected = np.column_stack((evidence.source_pixels, np.ones(len(optical)))) @ inverse_warp.T
    model = projected[:, :2] / projected[:, 2:3]
    if not np.allclose(model[evidence.crop_visible], evidence.model_pixels[evidence.crop_visible], rtol=1e-6, atol=1e-4):
        raise ValueError("ledger model pixels disagree with stock crop projection")


def interpolation_evidence(geometry, support, position, maximum_gap_m=MAX_OBSERVED_KNOT_GAP_M, method=None):
    """Reconstruct a corridor from observed boundary pairs, with separate QA.

    Raw pixel visibility is never filled in between paint markings. Only the
    selected observed knot coordinates define this spline. Full map coordinates
    are queried afterwards to validate its geometric agreement, not to derive
    its controls or replace its unsupported intervals.
    """
    left, right = np.asarray(geometry["left_xyz"], float), np.asarray(geometry["right_xyz"], float)
    center, arc = (left+right)*.5, np.asarray(geometry["arc_m"], float)
    n = len(arc)
    from .observed_corridor import METHOD, reconstruct, provenance
    if method==METHOD:
        complete=LaneReference(left,right,arc)
        timestamp=max(ref.timestamp_s for refs in support.evidence_frames for ref in refs)
        observed=reconstruct(complete,support,position,timestamp,maximum_gap_m)
        knots=observed.source_indices;chord=observed.s
        interval_errors=[]
        for interval,(first,last) in enumerate(zip(knots,knots[1:])):
            fraction=np.linspace(0,1,max(3,int(np.ceil((arc[last]-arc[first])/.1))+1))
            full_q=arc[first]+fraction*(arc[last]-arc[first])
            observed_q=chord[interval]+fraction*(chord[interval+1]-chord[interval])
            difference=np.concatenate([np.linalg.norm(getattr(observed,curve)(observed_q)-getattr(complete,curve)(full_q),axis=1)
                for curve in ('_left','_center','_right')])
            interval_errors.append(float(difference.max()))
        refs=observed.evidence_for_arc(chord[0],chord[-1])
        return dict(**provenance(observed),original_road_arc_m=arc[knots].tolist(),
            interval_chord_lengths_m=np.diff(chord).tolist(),interval_geometry_disagreement_m=interval_errors,
            interval_geometry_valid=(np.asarray(interval_errors)<=.05).tolist(),
            spline_evidence_frames=[vars(ref) for ref in refs])
    if method not in (None,'cubic_interpolation_of_rendered_paired_boundary_knots_no_hidden_map_derivatives'):
        raise ValueError('unknown observed boundary reconstruction method')
    indices = np.flatnonzero(support.supported[:n] & support.supported[n:])
    if len(indices) < 4:
        raise ValueError("fewer_than_four_paired_observed_knots")
    gaps = np.linalg.norm(np.diff(center[indices], axis=0), axis=1)
    runs = np.split(indices, np.flatnonzero(gaps > maximum_gap_m)+1)
    eligible = [run for run in runs if len(run) >= 4]
    if not eligible:
        raise ValueError("observed_paint_gaps_exceed_interpolation_limit")
    knots = min(eligible, key=lambda run: float(np.min(np.linalg.norm(center[run, :2]-np.asarray(position)[:2], axis=1))))
    chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center[knots], axis=0), axis=1))]
    observed = LaneReference(left[knots], right[knots], chord)
    complete = LaneReference(left, right, arc)
    interval_errors = []
    for interval, (first, last) in enumerate(zip(knots, knots[1:])):
        # Dense correspondence follows road arc, not closest points across a hairpin.
        fraction = np.linspace(0, 1, max(3, int(np.ceil((arc[last]-arc[first])/.1))+1))
        full_q = arc[first] + fraction*(arc[last]-arc[first])
        observed_q = chord[interval] + fraction*(chord[interval+1]-chord[interval])
        difference = np.concatenate([np.linalg.norm(getattr(observed, curve)(observed_q)-getattr(complete, curve)(full_q), axis=1)
                                     for curve in ("_left", "_center", "_right")])
        interval_errors.append(float(difference.max()))
    references = {(ref.frame_id, ref.camera, ref.timestamp_s): vars(ref) for knot in knots
                  for i in (knot, knot+n) for ref in support.evidence_frames[i]}
    return dict(source_indices=knots.tolist(), chord_arc_m=chord.tolist(),
        maximum_marking_gap_m=float(maximum_gap_m), interval_chord_lengths_m=np.diff(chord).tolist(),
        original_road_arc_m=arc[knots].tolist(), interval_geometry_disagreement_m=interval_errors,
        interval_geometry_valid=(np.asarray(interval_errors) <= .05).tolist(),
        spline_evidence_frames=list(references.values()),
        support_method="cubic_interpolation_of_rendered_paired_boundary_knots_no_hidden_map_derivatives")


def _replay_support(shard, manifest, frames, telemetry, geometry, candidates):
    archive_type = reference_module("frame_archive_v2").FrameArchiveReader
    readers = {}
    errors = []
    histories, results = EvidenceHistory(), {}
    arc = np.asarray(geometry["arc_m"], float)
    n = len(arc)
    ids = tuple([f"L{i}" for i in range(n)] + [f"R{i}" for i in range(n)])
    points = np.concatenate([geometry["left_xyz"], geometry["right_xyz"]])
    frame_lookup = {frame["frame_id"]: i for i, frame in enumerate(frames)}
    candidate_lookup = {int(item["index"]): item for item in candidates}
    candidate_indices = set(candidate_lookup)
    seen = set()
    if not manifest.get("camera_sensors"):
        errors.append("recorded_camera_transform_bindings_missing")
    try:
        for camera in ("narrow", "wide"):
            value = manifest["archives"][camera]
            readers[camera] = archive_type(_member(shard, value["path"]), expected_archive_sha256=value["sha256"])
            if len(readers[camera]) != len(frames) or readers[camera].frame_shape != (6, 128, 256) or readers[camera].dtype != np.uint8:
                raise ValueError("packed archives do not match dense frame metadata")
        evidence_path = _member(shard, manifest.get("evidence_file", "evidence.jsonl"))
        with (gzip.open(evidence_path, "rt", encoding="utf-8") if evidence_path.suffix == ".gz" else evidence_path.open(encoding="utf-8")) as stream:
            for line in stream:
                if not line.strip():
                    continue
                doc = json.loads(line)
                ref = EvidenceRef(**doc["reference"])
                index = frame_lookup.get(ref.frame_id)
                if index is None or abs(ref.timestamp_s - frames[index]["timestamp"]) > 1e-6:
                    raise ValueError("evidence frame is not an authenticated recorded timestamp")
                if tuple(doc["point_ids"]) != ids or not np.allclose(doc["arc_m"], np.tile(arc, 2), atol=1e-8):
                    raise ValueError("evidence point identities disagree with recorded paired boundaries")
                for key in ("source_visible", "crop_visible"):
                    if not all(isinstance(v, bool) for v in doc[key]):
                        raise ValueError("evidence visibility values must be boolean")
                kwargs = {field.name: np.asarray(doc[field.name], dtype=bool if field.name in ("source_visible", "crop_visible") else float)
                          for field in fields(FrameEvidence) if field.name not in ("reference", "point_ids", "sample_depth_basis")
                          and doc.get(field.name) is not None}
                kwargs['sample_depth_basis']=doc.get('sample_depth_basis','projected_point')
                if kwargs['sample_depth_basis'] not in ('projected_point','local_surface_plane_at_exact_sample_pixel'):
                    raise ValueError('unrecognized depth sampling basis')
                evidence = FrameEvidence(ref, ids, **kwargs)
                if manifest.get("camera_sensors"):
                    _verify_projection(evidence, manifest, telemetry, index, points)
                _verify_pixel_claims(evidence, readers[ref.camera][index], telemetry["position_world"][index], points)
                histories.admit(evidence)
                seen.add((index, ref.camera))
                if index in candidate_indices and (index, "narrow") in seen and (index, "wide") in seen:
                    distance = np.tile(arc, 2) - telemetry["route_s"][index]
                    speed = float(telemetry["speed"][index])
                    needed = (distance >= 0) & (distance <= max(3., speed * max(.5, frames[index]["action_t"][0])))
                    support = histories.support(ref.frame_id, ref.timestamp_s, ids, needed,
                        np.maximum(0, distance) / max(speed, 1), [ids[i] for i in np.flatnonzero(needed)], distance_m=distance)
                    if candidate_lookup[index].get("observable_reference") is not None:
                        try:
                            interpolation = interpolation_evidence(geometry, support, telemetry["position_world"][index],
                                method=candidate_lookup[index]['observable_reference'].get('support_method'))
                            knot_mask = np.zeros(2*n, bool)
                            if 'source_point_indices' in interpolation:
                                knot_mask[interpolation['source_point_indices']]=True
                            else:
                                knot_mask[interpolation["source_indices"]] = True
                                knot_mask[np.asarray(interpolation["source_indices"])+n] = True
                            support = histories.support(ref.frame_id, ref.timestamp_s, ids, knot_mask,
                                np.maximum(0, distance) / max(speed, 1), [ids[i] for i in np.flatnonzero(knot_mask)], distance_m=distance)
                        except (ValueError, TypeError):
                            pass  # This candidate remains diagnostic; other timestamps remain inspectable.
                    results[index] = support
        if len(seen) != 2 * len(frames):
            errors.append("incomplete_dense_camera_evidence")
    except (KeyError, ValueError, TypeError, OSError) as error:
        errors.append("pixel_evidence_verification_failed:" + str(error))
        results = {}  # A corrupted ledger cannot lend support to earlier targets.
    finally:
        for reader in readers.values():
            reader.close()
    return results, errors


def select_phase_indices(indices, route_s, phase_ranges=None, limit=20):
    """Four distinct samples per reviewed maneuver phase, preserving the 2 Hz grid."""
    values = np.asarray(sorted(set(map(int, indices))), int)
    if not len(values):
        return values, {}, ["no_supported_candidate_labels"]
    errors = []
    if not isinstance(phase_ranges, dict) or set(phase_ranges) != set(PHASES):
        chosen = values[np.unique(np.rint(np.linspace(0, len(values)-1, min(limit, len(values)))).astype(int))]
        return chosen, {int(i): "progress_only_unreviewed_phase" for i in chosen}, ["maneuver_phase_semantics_unreviewed"]
    chosen, phases = [], {}
    for phase in PHASES:
        lo, hi = map(float, phase_ranges[phase])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("maneuver phase range must be finite and increasing")
        eligible = np.asarray([i for i in values if lo <= route_s[i] < hi or (phase == "recovery" and route_s[i] == hi)], int)
        eligible = np.asarray([i for i in eligible if i not in phases], int)
        if len(eligible) < 4:
            errors.append(f"insufficient_phase_candidates:{phase}:{len(eligible)}/4")
        positions = np.unique(np.rint(np.linspace(0, len(eligible)-1, min(4, len(eligible)))).astype(int)) if len(eligible) else []
        for index in eligible[positions]:
            chosen.append(int(index)); phases[int(index)] = phase
    return np.asarray(sorted(chosen), int), phases, errors


def _plan_support(geometry, telemetry, index, support, interpolation=None):
    """Map each time sample to paired lane knots and a continuous observed path.

    Geometry after the first unsupported interpolation interval stays masked.
    A global map spline is never used to fill a gap in admitted observations.
    """
    arc = np.asarray(geometry["arc_m"], float)
    paired = support.supported[:len(arc)] & support.supported[len(arc):]
    times = telemetry["timestamp"]
    query = times[index] + PLAN_TIMES
    progress = np.interp(query, times, telemetry["route_s"])
    if interpolation is not None:
        knots = np.asarray(interpolation["original_road_arc_m"])
        interval_valid = np.asarray(interpolation["interval_geometry_valid"], bool)
        mask = np.zeros(33, bool)
        evidence, broken = [], False
        previous = float(telemetry["route_s"][index])
        for j, coordinate in enumerate(progress):
            lo = min(len(knots)-2, max(0, int(np.searchsorted(knots, min(previous, coordinate), side="right"))-1))
            hi = min(len(knots)-2, max(lo, int(np.searchsorted(knots, max(previous, coordinate), side="left"))-1))
            available = (query[j] <= times[-1]+1e-8 and knots[0] <= min(previous, coordinate)
                         and max(previous, coordinate) <= knots[-1] and interval_valid[lo:hi+1].all())
            broken |= not available
            mask[j] = not broken
            evidence.append(interpolation["spline_evidence_frames"] if mask[j] else [])
            previous = float(coordinate)
        return mask, evidence
    mask = np.zeros(33, bool)
    evidence = []
    previous = float(telemetry["route_s"][index])
    broken = False
    for j, coordinate in enumerate(progress):
        # Include both bracketing boundary samples, including the path swept
        # since the preceding target time (not only an isolated endpoint).
        lo = max(0, int(np.searchsorted(arc, min(previous, coordinate), side="right")) - 1)
        hi = min(len(arc)-1, max(lo+1, int(np.searchsorted(arc, max(previous, coordinate), side="left"))))
        available = (query[j] <= times[-1] + 1e-8 and arc[0] <= coordinate <= arc[-1]
                     and hi > lo and paired[lo:hi+1].all())
        broken |= not available
        mask[j] = not broken
        refs = []
        if mask[j]:
            refs = [ref for k in range(lo, hi+1) for side in (k, k+len(arc)) for ref in support.evidence_frames[side]]
        unique = {(ref.frame_id, ref.camera, ref.timestamp_s): ref for ref in refs}
        evidence.append([vars(ref) for ref in unique.values()])
        previous = float(coordinate)
    return mask, evidence


def _load_command_provenance(shard, manifest, frames, telemetry):
    """Load a hash-bound dense transition ledger; legacy captures return None."""
    name = manifest.get("command_provenance_file", "command-provenance.jsonl.gz")
    path = _member(shard, name)
    if not path.exists():
        if name in manifest.get("files", {}) or "command_provenance_file" in manifest:
            raise ValueError("declared dense command provenance is missing")
        return None
    if manifest.get("files", {}).get(name) != sha256_file(path):
        raise ValueError("dense command provenance hash missing or mismatched")
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != len(frames):
        raise ValueError("dense command provenance must cover every recorded frame")
    for index, (record, frame) in enumerate(zip(records, frames)):
        if (record.get("index") != index or record.get("frame_id") != frame["frame_id"] or
            not np.isclose(record.get("timestamp", np.nan), frame["timestamp"], atol=1e-7, rtol=0) or
            not isinstance(record.get("command_accepted"), bool) or
            record["command_accepted"] != bool(telemetry["command_accepted"][index])):
            raise ValueError("dense command provenance is not aligned with physical telemetry")
    return records


def causal_plan_support(frames, telemetry, index, geometry_mask, support, provenance):
    """Admit recorded states only before the first future-information command.

    A command recorded at k acts after state k. Position/velocity/orientation at
    an exact tick depend on preceding transitions. Interpolated samples depend
    on their right endpoint; Hermite acceleration and SLERP rates also depend
    on the next endpoint at exact ticks, including the initial state.
    """
    times = np.asarray(telemetry["timestamp"], float)
    points = np.asarray(geometry_mask, bool)
    if points.shape != (33,) or len(support.supported) % 2:
        raise ValueError("causal plans require 33 geometry flags and paired boundary support")
    count = len(support.supported) // 2
    admitted = set(np.flatnonzero(support.supported[:count] & support.supported[count:]).tolist())
    barrier, reason = index, "dense_command_provenance_missing"
    if provenance is not None:
        barrier, reason = len(times)-1, "all_recorded_transitions_use_decision_visible_knots"
        last_needed = min(len(times)-2, int(np.searchsorted(times, times[index]+PLAN_TIMES[-1], side="right"))-1)
        for step in range(index, last_needed+1):
            record = provenance[step]
            independent='source_point_indices' in record
            knots = record.get('source_point_indices') if independent else record.get("source_indices")
            permitted=set(np.flatnonzero(support.supported).tolist()) if independent else admitted
            point_count=2*count if independent else count
            references = record.get("evidence_frames")
            failure = None
            if not record.get("command_accepted"):
                failure = "future_command_not_accepted"
            elif (not isinstance(knots, list) or not knots or
                  any(type(value) is not int or value < 0 or value >= point_count for value in knots) or
                  len(set(knots)) != len(knots)):
                failure = "future_command_knot_provenance_invalid"
            elif not set(knots).issubset(permitted):
                failure = "future_command_introduces_unseen_knots"
            elif not isinstance(references, list) or not references:
                failure = "future_command_evidence_missing"
            else:
                for ref in references:
                    frame_id = ref.get("frame_id")
                    evidence_index = frame_id-frames[0]["frame_id"] if type(frame_id) is int else -1
                    if (not max(0, step-96) <= evidence_index <= step or
                        ref.get("camera") not in ("narrow", "wide") or
                        not np.isclose(ref.get("timestamp_s", np.nan), times[evidence_index], atol=1e-7, rtol=0)):
                        failure = "future_command_evidence_not_causal_or_aligned"
                        break
            if failure:
                barrier, reason = step, failure
                break
    query = times[index]+PLAN_TIMES
    # Use the same floating-point query as dense_to_plan. Even a tiny positive
    # interpolation weight on a later state requires its transition provenance.
    right = np.searchsorted(times, query, side="left")
    segment = np.minimum(np.searchsorted(times, query, side="right")-1, len(times)-2)
    available = query <= times[-1]+1e-9
    state_ok = points & available & (right <= barrier)
    derivative_ok = points & available & (segment+1 <= barrier)
    result = np.zeros((33,15), bool)
    result[:,0:6] = state_ok[:,None]
    result[:,9:12] = state_ok[:,None]
    result[:,6:9] = derivative_ok[:,None]
    result[:,12:15] = derivative_ok[:,None]
    return result, dict(method="dense_recorded_transition_provenance",
        command_ledger_present=provenance is not None, first_unsupported_transition_index=int(barrier),
        supported_state_horizon_s=float(min(10., max(0., times[barrier]-times[index]))), reason=reason,
        initial_interpolation_derivatives_supported=bool(derivative_ok[0]))


def build_labels(shard_path, *, dataset_root=None):
    """Serialize label and manifest updates against all other shard writers."""
    with shard_postprocess_lock(shard_path):
        return _build_labels_unlocked(shard_path, dataset_root=dataset_root)


def _build_labels_unlocked(shard_path, *, dataset_root=None):
    shard = Path(shard_path).resolve()
    manifest_path = shard / "manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") == "accepted":
        raise FileExistsError("accepted shards are immutable")
    root = _root_for(shard, dataset_root)
    errors = []
    preprocessing = manifest.get("preprocessing", {})
    current_preprocessing = StockPreprocessor().contract()
    for key in ("crops", "focal_pixels", "model_to_camera", "packed_order", "rgb_producer_function"):
        if preprocessing.get(key) != current_preprocessing[key]:
            errors.append("stock_preprocessing_contract_mismatch:" + key)
    for name, binding in current_preprocessing["reference_sources"].items():
        if preprocessing.get("reference_sources", {}).get(name, {}).get("sha256") != binding["sha256"]:
            errors.append("preprocessing_source_hash_mismatch:" + name)
    for name in (manifest.get("frames_file", "frames.jsonl"), "telemetry.npz", "geometry.json",
                 manifest.get("candidates_file", "candidates.json"), manifest.get("evidence_file", "evidence.jsonl"), "depth-fixture.json"):
        expected = manifest.get("files", {}).get(name)
        if not expected or not (shard / name).is_file() or sha256_file(shard / name) != expected:
            errors.append("capture_artifact_hash_missing_or_mismatch:" + name)
    if any(reason.startswith("capture_artifact_hash") for reason in errors):
        raise ValueError(";".join(errors))
    frames = [json.loads(line) for line in (shard / manifest["frames_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids, times = validate_frame_metadata(frames)
    with np.load(shard / "telemetry.npz", allow_pickle=False) as archive:
        telemetry = {key: archive[key] for key in archive.files}
    if not np.array_equal(ids, telemetry["frame_id"]) or not np.allclose(times, telemetry["timestamp"], rtol=0, atol=1e-6):
        raise ValueError("telemetry is not aligned with recorded input frames")
    command_provenance = _load_command_provenance(shard, manifest, frames, telemetry)
    geometry, candidates = _json(shard / "geometry.json"), _json(_member(shard, manifest.get("candidates_file", "candidates.json")))
    reference = LaneReference(geometry["left_xyz"], geometry["right_xyz"], geometry["arc_m"])
    hint = None
    for index, position in enumerate(telemetry["position_world"]):
        progress, _ = reference.project(position[:2], hint_s=hint)
        hint = progress
        if abs(progress - telemetry["route_s"][index]) > .02:
            raise ValueError("recorded route progress is inconsistent with physical position")
    seen = set()
    for item in candidates:
        index = item["index"]
        if not isinstance(index, int) or index in seen or index % 10 or index < 100 or index > len(frames)-201:
            raise ValueError("candidate index is duplicate, off-grid, or lacks required retention")
        seen.add(index)
    calibration, calibration_errors, calibration_binding = _calibration_status(manifest, root)
    depth_errors = _depth_status(shard)
    support_by_index, evidence_errors = _replay_support(shard, manifest, frames, telemetry, geometry, candidates)
    errors += calibration_errors + depth_errors + evidence_errors
    qualified_indices, provisional, interpolations = [], [], {}
    candidate_reasons = {}
    for candidate in candidates:
        index = candidate["index"]
        support = support_by_index.get(index)
        reasons = []
        if support is None:
            reasons.append("no_verified_evidence_at_decision")
        else:
            if candidate.get("support") is None or candidate["support"].get("supported") != support.supported.tolist():
                reasons.append("captured_support_disagrees_with_replayed_evidence")
            if not support.action_valid or support.category == "diagnostic_ambiguous":
                reasons.append("immediate_action_has_unsupported_geometry")
            provisional.append(index)
            recorded_interpolation = candidate.get("observable_reference")
            if recorded_interpolation is None:
                reasons.append("observed_interpolation_provenance_missing")
            else:
                try:
                    interpolation = interpolation_evidence(geometry, support, telemetry["position_world"][index],
                        method=recorded_interpolation.get('support_method'))
                    interpolations[index] = interpolation
                    if (recorded_interpolation.get("source_indices") != interpolation["source_indices"] or
                        recorded_interpolation.get('source_point_indices') != interpolation.get('source_point_indices') or
                        not np.allclose(recorded_interpolation.get("chord_arc_m", []), interpolation["chord_arc_m"], atol=1e-7) or
                        recorded_interpolation.get("support_method") != interpolation["support_method"] or
                        recorded_interpolation.get("maximum_marking_gap_m") != MAX_OBSERVED_KNOT_GAP_M):
                        reasons.append("observed_interpolation_does_not_match_admitted_knots")
                    current_s = float(telemetry["route_s"][index])
                    action_end_s = current_s + float(telemetry["speed"][index])*frames[index]["action_t"][0]
                    if current_s < interpolation["original_road_arc_m"][0] or action_end_s > interpolation["original_road_arc_m"][-1]:
                        reasons.append("immediate_action_requires_extrapolation_beyond_observed_corridor")
                    if not all(interpolation["interval_geometry_valid"]):
                        reasons.append("observed_interpolation_geometry_disagreement_above_five_cm")
                except (ValueError, TypeError) as error:
                    reasons.append("observed_interpolation_invalid:"+str(error))
            command_refs = candidate.get("command_evidence", [])
            admitted_refs = {(ref.frame_id, ref.timestamp_s, ref.camera)
                             for point_refs in support.evidence_frames for ref in point_refs}
            if not command_refs:
                reasons.append("expert_command_evidence_missing")
            else:
                for ref in command_refs:
                    key = (ref.get("frame_id"), ref.get("timestamp_s"), ref.get("camera"))
                    if key not in admitted_refs:
                        reasons.append("expert_command_uses_unadmitted_frame_evidence")
                        break
        if calibration is None or calibration_errors:
            reasons.append("calibration_invalid")
        else:
            speed = float(telemetry["speed"][index])
            curvature = float(telemetry["command_curvature"][index])
            reasons.extend(command_domain_errors(calibration, speed, curvature, frames[index]["action_t"][0]))
        if not bool(telemetry["command_accepted"][index]) or candidate.get("command_valid") is not True:
            reasons.append("expert_command_not_accepted")
        if not np.isfinite(telemetry["command_curvature"][index]):
            reasons.append("expert_command_nonfinite")
        if depth_errors or evidence_errors or any(reason.startswith(("stock_preprocessing", "preprocessing_source")) for reason in errors):
            reasons.append("rendered_evidence_invalid")
        if manifest.get("family") == "intersection_turn":
            direction = manifest.get("direction", geometry.get("section", {}).get("direction"))
            desire_index = {"left": 1, "right": 2}.get(direction)
            intent_indices = manifest.get("intent_required_indices")
            intent_range = manifest.get("intent_required_range_s")
            spatial_intent = manifest.get("intent_required_route_range_m")
            if (isinstance(spatial_intent,list) and len(spatial_intent)==2 and
                np.isfinite(spatial_intent).all() and spatial_intent[0]<spatial_intent[1]):
                requires_intent=spatial_intent[0]<=telemetry["route_s"][index]<=spatial_intent[1]
            elif isinstance(intent_indices, list) and all(isinstance(value, int) for value in intent_indices):
                requires_intent = index in intent_indices
            elif isinstance(intent_range, list) and len(intent_range) == 2 and np.isfinite(intent_range).all() and intent_range[0] <= intent_range[1]:
                requires_intent = intent_range[0] <= times[index]-times[0] <= intent_range[1]
            else:
                requires_intent = True
                reasons.append("intersection_branch_decision_scope_unreviewed")
            if desire_index is None and requires_intent:
                reasons.append("intersection_turn_direction_missing")
            elif requires_intent:
                # A held desire emits only one pulse, so presence in the raw
                # desire vector alone does not prove timely route intent.
                rising = [i for i in range(max(0, index-99), index)
                          if frames[i]["desire"][desire_index] == 1 and
                          (i == 0 or frames[i-1]["desire"][desire_index] == 0)]
                if not rising:
                    reasons.append("intersection_intent_pulse_missing_before_decision")
        candidate_reasons[index] = reasons
        if not reasons:
            qualified_indices.append(index)
    pool = qualified_indices if len(qualified_indices) >= 20 else provisional
    indices, phases, phase_errors = select_phase_indices(pool, telemetry["route_s"],
                                                        manifest.get("maneuver_phase_ranges") or geometry.get("maneuver_phase_ranges"))
    errors += phase_errors
    point_masks = np.zeros((len(indices), 33), bool)
    causal_masks = np.zeros((len(indices), 33, 15), bool)
    segment_evidence, causality = [], []
    for row, index in enumerate(indices):
        point_masks[row], refs = _plan_support(geometry, telemetry, int(index), support_by_index[int(index)], interpolations.get(int(index)))
        causal_masks[row], proof = causal_plan_support(frames, telemetry, int(index), point_masks[row],
                                                     support_by_index[int(index)], command_provenance)
        segment_evidence.append([ref if causal_masks[row,j].any() else [] for j,ref in enumerate(refs)])
        causality.append(proof)
    if len(indices):
        plans, plan_masks = dense_to_plan(times, telemetry["position_world"], telemetry["velocity_world"],
                                        telemetry["rotation_world"], indices, causal_masks)
    else:
        plans, plan_masks = np.empty((0, 33, 15), np.float32), np.empty((0, 33, 15), bool)
    lateral_mask = np.asarray([not candidate_reasons[int(i)] for i in indices], bool)
    speeds = telemetry["speed"][indices].astype(np.float32)
    curvatures = telemetry["command_curvature"][indices].astype(np.float32)
    lateral_targets = np.full(len(indices), np.nan, np.float32)
    finite_action = np.isfinite(speeds) & (speeds >= 0) & np.isfinite(curvatures)
    lateral_targets[finite_action] = lateral_action_target(speeds[finite_action], curvatures[finite_action])
    arrays = dict(label_indices=indices, plan_targets=plans, plan_mask=plan_masks,
        current_speed=speeds, commanded_curvature=curvatures,
        lateral_targets=lateral_targets, lateral_mask=lateral_mask)
    temporary = shard / "labels.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(shard / "labels.npz")
    records = []
    for row, index in enumerate(indices):
        support = support_by_index[int(index)]
        candidate = next(item for item in candidates if item["index"] == index)
        action_refs = candidate.get("command_evidence", []) if lateral_mask[row] else []
        records.append(dict(accepted=False, split=manifest.get("split", "development"),
            family=manifest.get("family"), group_id=manifest.get("geometry_group_id"),
            episode_id=manifest["episode_id"], label_index=int(index), timestamp_s=float(times[index]),
            candidate_grid_origin_s=float(times[0]), appearance=manifest.get("appearance", 0),
            procedural=bool(manifest.get("procedural", False)),
            direction=manifest.get("direction", geometry.get("section", {}).get("direction", "unknown")),
            phase=phases[int(index)], visibility_category=support.category, visibility=support.compact(),
            interpolated_corridor=interpolations.get(int(index)),
            action_valid=bool(lateral_mask[row]), action_t=frames[index]["action_t"],
            action_evidence_frames=action_refs, plan_segment_evidence_frames=segment_evidence[row],
            observable_support_mask=plan_masks[row].any(axis=1).tolist(),
            plan_element_support_mask=plan_masks[row].tolist(), plan_causality=causality[row],
            qa_passed=False, reviewed=False,
            rejection_reasons=candidate_reasons[int(index)]))
    if len(indices) != 20 or not lateral_mask.all():
        errors.append(f"supported_action_label_quota:{int(lateral_mask.sum())}/20")
    report = dict(schema_version=1, label_count=len(indices), lateral_label_count=int(lateral_mask.sum()),
        supported_plan_elements=int(plan_masks.sum()), candidate_count=len(candidates),
        command_provenance_present=command_provenance is not None,
        plan_causality_restrictions=dict(__import__("collections").Counter(item["reason"] for item in causality)),
        diagnostic_candidate_count=sum(bool(value) for value in candidate_reasons.values()),
        build_errors=sorted(set(errors)), calibration_binding=calibration_binding,
        candidate_rejections={str(key): value for key, value in candidate_reasons.items()},
        status="labels_built_not_admitted", passed=False)
    write_json(shard / "labels-report.json", report)
    manifest.update(label_records=records, accepted_label_count=0, labels_file="labels.npz")
    manifest["files"].update({name: sha256_file(shard / name) for name in ("labels.npz", "labels-report.json")})
    write_json(manifest_path, manifest)
    return report


def verify_qualification(dataset_root, manifest):
    """Require hash-bound executed evidence, source inventory and frozen splits.

    A dictionary of `gates: true` has no admission meaning. Existing XML test
    evidence must contain actual passing testcases; JSON reports must carry
    measured records in addition to their result field.
    """
    root = Path(dataset_root).resolve()
    errors, bindings = [], {}
    try:
        qualification = _json(root / "qualification.json")
        config = _json(root / "config.json")
        if qualification.get("schema_version") != 1 or config.get("frozen") is not True:
            errors.append("configuration_not_frozen")
        expected = Config().digest()
        if any(value != expected for value in (qualification.get("config_sha256"), config.get("sha256"), manifest.get("config_sha256"))):
            errors.append("configuration_binding_mismatch")
        source_map = qualification.get("sources", {})
        normalized = {(Path(name) if Path(name).is_absolute() else PROJECT / name).resolve(): digest for name, digest in source_map.items()}
        # Re-enumerate at verification time so a module added after this process
        # started cannot evade the qualification source seal.
        current_sources = tuple(sorted((PROJECT / "lane_dataset").glob("*.py")))
        if len(normalized) != len(source_map) or set(normalized) != {source.resolve() for source in current_sources}:
            errors.append("qualification_source_inventory_mismatch")
        for source in current_sources:
            if normalized.get(source.resolve()) != sha256_file(source):
                errors.append("qualification_source_binding_mismatch:" + source.name)
        for gate in GATES:
            path = _binding(qualification.get("evidence", {}).get(gate), root)
            bindings[gate] = {"path": str(path), "sha256": sha256_file(path)}
            if path.suffix.lower() == ".xml":
                tree = ET.parse(path).getroot()
                cases = list(tree.iter("testcase"))
                if not cases or any(list(case.iter(tag)) for case in cases for tag in ("failure", "error", "skipped")):
                    errors.append("qualification_test_evidence_failed:" + gate)
                patterns = {
                    "runtime_action": ("test_lateral_target_through_actual_decoder", "test_actual_runtime_angle_controller_and_vehicle_model_roundtrip"),
                    "hidden_road_perturbation": ("test_hidden_geometry_cannot_influence_admitted_derivatives_or_command",),
                    "split_overlap": ("test_recomputed_overlap_catches_manipulated_group_ids",),
                    "preprocessing": ("test_luma_visibility_sampler_is_exact_stock_packed_luma", "test_stock_warps_and_plane_order"),
                }.get(gate)
                if patterns is None or any(not any(case.get("name", "").startswith(pattern) for case in cases) for pattern in patterns):
                    errors.append("qualification_test_does_not_establish_physical_gate:" + gate)
            else:
                evidence = _json(path)
                result = any(evidence.get(key) is True for key in ("passed", "qualified", "gate_pass"))
                measured = any(isinstance(evidence.get(key), (dict, list)) and bool(evidence[key]) for key in
                               ("measurements", "warp_cases", "cases", "tests", "metrics", "validation_reports"))
                if not result or not measured or evidence.get("errors") or evidence.get("failures"):
                    errors.append("qualification_measurement_evidence_failed:" + gate)
        split_path = _binding(qualification.get("split_binding"), root)
        split = _json(split_path)
        if split.get("splits_frozen") is not True:
            errors.append("split_assignments_not_frozen")
        from .splits import GeometryGroup, GeometrySection, RoadInterval, SplitAssignment, audit_split_overlap
        groups = []
        for value in split["groups"]:
            sections = [GeometrySection(**{**section, "road_intervals": tuple(RoadInterval(**interval) for interval in section["road_intervals"])}) for section in value["sections"]]
            groups.append(GeometryGroup(value["group_id"], tuple(sections)))
        assignments = [SplitAssignment(**value) for value in split["assignments"]]
        errors += audit_split_overlap(groups, assignments)
        matches = [assignment for assignment in assignments if assignment.group_id == manifest.get("geometry_group_id")]
        if len(matches) != 1 or matches[0].split != manifest.get("split") or matches[0].family != manifest.get("family"):
            errors.append("shard_does_not_match_frozen_split_assignment")
        sealed_path=qualification.get('sealed_path')
        if sealed_path is not None:
            immutable=_member(root,sealed_path)
            if _json(immutable)!=qualification:
                errors.append('immutable_qualification_version_mismatch')
        else:immutable=root/'qualification.json'
        bindings["qualification"] = {"path": str(immutable), "sha256": sha256_file(immutable)}
        bindings["split"] = {"path": str(split_path), "sha256": sha256_file(split_path)}
    except (KeyError, ValueError, TypeError, OSError, ET.ParseError) as error:
        errors.append("qualification_evidence_invalid:" + str(error))
    return {"passed": not errors, "errors": sorted(set(errors)), "bindings": bindings}


def validate_capture_envelope(shard, manifest, telemetry):
    """Independently check raw recording seam and bound traffic coverage."""
    shard = Path(shard)
    errors, bindings, measured = [], {}, {}
    if "error" in manifest:
        errors.append("capture_error_recorded")
    bootstrap_count = None
    try:
        name = "bootstrap-telemetry.npz"
        path = _binding(dict(path=name,sha256=manifest.get("files",{}).get(name)),shard)
        bindings["bootstrap_telemetry"] = dict(path=str(path),sha256=sha256_file(path))
        with np.load(path,allow_pickle=False) as archive:
            raw = {key:archive[key] for key in ("frame_id","timestamp","rotation_world","acceleration_world")}
        bootstrap_count = len(raw["timestamp"])
        if (bootstrap_count < 1 or raw["frame_id"].shape != (bootstrap_count,) or
            raw["frame_id"].dtype.kind not in "iu" or
            any(raw[key].shape != (bootstrap_count,3) for key in ("rotation_world","acceleration_world")) or
            not all(np.isfinite(value).all() for value in raw.values()) or
            np.any(np.diff(raw["frame_id"]) != 1) or
            not np.allclose(np.diff(raw["timestamp"]),.05,rtol=0,atol=1e-6)):
            raise ValueError("bootstrap raw frames are missing, unordered, nonfinite, or not 20 Hz")
        bootstrap = manifest.get("bootstrap",{})
        if bootstrap.get("ready") is not True or bootstrap.get("frames") != bootstrap_count:
            errors.append("bootstrap_readiness_or_frame_count_mismatch")
        first_frame, last_frame = int(telemetry["frame_id"][0]),int(raw["frame_id"][-1])
        last_time, first_time = float(raw["timestamp"][-1]),float(telemetry["timestamp"][0])
        if (bootstrap.get("last_frame_id") != last_frame or
            not np.isclose(bootstrap.get("last_timestamp",np.nan),last_time,rtol=0,atol=1e-7)):
            errors.append("bootstrap_last_snapshot_binding_mismatch")
        interval = first_time-last_time
        acceleration = []
        for rotation, vector in ((raw["rotation_world"][-1],raw["acceleration_world"][-1]),
                                 (telemetry["rotation_world"][0],telemetry["acceleration_world"][0])):
            rotation, vector = np.asarray(rotation,float),np.asarray(vector,float)
            if rotation.shape != (3,) or vector.shape != (3,) or not np.isfinite([*rotation,*vector]).all():
                raise ValueError("recording seam requires finite physical rotation and acceleration")
            acceleration.append(float(-np.sin(rotation[2])*vector[0]+np.cos(rotation[2])*vector[1]))
        jerk = (acceleration[1]-acceleration[0])/interval if interval > 0 else None
        limits = QualityConfig()
        if first_frame != last_frame+1 or not np.isclose(interval,.05,atol=1e-6,rtol=0):
            errors.append("bootstrap_recording_seam_missing_duplicate_or_non20hz_frame")
        if max(map(abs,acceleration)) > limits.lateral_acceleration_limit_mps2:
            errors.append("bootstrap_recording_seam_lateral_acceleration_above_limit")
        if jerk is None or abs(jerk) > limits.lateral_jerk_limit_mps3:
            errors.append("bootstrap_recording_seam_lateral_jerk_above_limit")
        measured["recording_seam"] = dict(last_bootstrap_frame_id=last_frame,first_recorded_frame_id=first_frame,
            interval_s=interval,lateral_acceleration_mps2=acceleration,lateral_jerk_mps3=jerk,
            maximum_lateral_acceleration_mps2=limits.lateral_acceleration_limit_mps2,
            maximum_lateral_jerk_mps3=limits.lateral_jerk_limit_mps3)
        declared = bootstrap.get("recording_seam",{})
        if declared.get("passed") is not True or declared.get("errors") != []:
            errors.append("bootstrap_recording_seam_missing_or_declared_failed")
        try:
            if (declared.get("last_bootstrap_frame_id") != last_frame or declared.get("first_recorded_frame_id") != first_frame or
                not np.isclose(declared.get("interval_s",np.nan),interval,rtol=0,atol=1e-7) or
                np.asarray(declared.get("lateral_acceleration_mps2",[])).shape != (2,) or
                not np.allclose(declared["lateral_acceleration_mps2"],acceleration,rtol=0,atol=1e-7) or
                jerk is None or not np.isclose(declared.get("lateral_jerk_mps3",np.nan),jerk,rtol=0,atol=1e-7)):
                errors.append("bootstrap_recording_seam_measurement_mismatch")
        except (KeyError,TypeError,ValueError):
            errors.append("bootstrap_recording_seam_measurement_mismatch")
    except (KeyError,ValueError,TypeError,OSError) as error:
        errors.append("bootstrap_recording_seam_evidence_invalid:"+str(error))
    count = manifest.get("traffic_count",0)
    if type(count) is not int or count < 0:
        errors.append("traffic_count_invalid")
    elif count == 0:
        if manifest.get("traffic") or (shard/"traffic-report.json").exists():
            errors.append("traffic_evidence_present_but_declared_count_zero")
    else:
        try:
            name = "traffic-report.json"
            path = _binding(dict(path=name,sha256=manifest.get("files",{}).get(name)),shard)
            bindings["traffic_report"] = dict(path=str(path),sha256=sha256_file(path))
            report = _json(path)
            if manifest.get("traffic") != report:
                errors.append("traffic_manifest_report_mismatch")
            if report.get("requested_count") != count or report.get("spawned_count") != count or len(report.get("spawn_records",[])) != count:
                errors.append("traffic_requested_spawned_count_mismatch")
            for key, value in (("physics",True),("hybrid_physics",False),("synchronous",True),
                               ("automatic_lane_change",False),("passed",True)):
                if report.get(key) is not value:
                    errors.append("traffic_configuration_or_result_failed:"+key)
            for key in ("collisions","cleanup_errors","reasons"):
                if report.get(key) != []:
                    errors.append("traffic_"+key+"_missing_or_nonempty")
            expected_frames = None if bootstrap_count is None else bootstrap_count+len(telemetry["timestamp"])
            if expected_frames is None or report.get("validated_frames") != expected_frames:
                errors.append("traffic_does_not_cover_every_owned_camera_tick")
            for key, minimum in (("minimum_corridor_clearance_m",0.),("minimum_body_clearance_m",.15)):
                value = report.get(key)
                if not isinstance(value,(int,float)) or not np.isfinite(value) or value < minimum:
                    errors.append("traffic_clearance_missing_or_failed:"+key)
            measured["traffic"] = report
        except (KeyError,ValueError,TypeError,OSError) as error:
            errors.append("traffic_evidence_invalid:"+str(error))
    return dict(passed=not errors,errors=sorted(set(errors)),bindings=bindings,measurements=measured)


def qualify_candidate(shard_path, *, dataset_root=None):
    """Rebuild labels, recompute physical gates, and check review provenance."""
    with shard_postprocess_lock(shard_path):
        return _qualify_candidate_unlocked(shard_path, dataset_root=dataset_root)


def _qualify_candidate_unlocked(shard_path, *, dataset_root=None):
    shard = Path(shard_path).resolve()
    root = _root_for(shard, dataset_root)
    built = _build_labels_unlocked(shard, dataset_root=root)
    manifest = _json(shard / "manifest.json")
    errors = list(built["build_errors"])
    qualification = verify_qualification(root, manifest)
    qualification["bindings"].update({"calibration_"+key: value for key, value in built["calibration_binding"].items()})
    errors += qualification["errors"]
    captured_sources=manifest.get("source_hashes",{})
    for source in sorted((PROJECT/"lane_dataset").glob("*.py")):
        captured=shard/"executed-sources"/source.name
        if (captured_sources.get(str(source))!=sha256_file(source) or not captured.is_file()
            or sha256_file(captured)!=sha256_file(source)):
            errors.append("capture_source_not_qualified:"+source.name)
    physical, envelope = None, None
    try:
        with np.load(shard / "telemetry.npz", allow_pickle=False) as archive:
            t = {key: archive[key] for key in archive.files}
        envelope = validate_capture_envelope(shard,manifest,t)
        errors += envelope["errors"]
        qualification["bindings"].update(envelope["bindings"])
        geometry = _json(shard / "geometry.json")
        reference = LaneReference(geometry["left_xyz"], geometry["right_xyz"], geometry["arc_m"])
        yaw = t["rotation_world"][:, 2]
        accel = t["acceleration_world"]
        lateral = -np.sin(yaw)*accel[:, 0] + np.cos(yaw)*accel[:, 1]
        target = manifest.get("intended_progress_m")
        if not isinstance(target, (int, float)) or target <= 0:
            errors.append("intended_maneuver_progress_not_bound")
            target = float(geometry["arc_m"][-1])
        dimensions = manifest["vehicle_dimensions"]
        height = dimensions.get("height")
        bbox_center = np.asarray(dimensions.get("bbox_center"), dtype=float)
        if (not isinstance(height, (int, float)) or not np.isfinite(height) or height <= 0 or
            bbox_center.shape != (3,) or not np.isfinite(bbox_center).all()):
            errors.append("full_3d_vehicle_bounding_box_missing_or_invalid")
            # Retain a clearly labelled planar diagnostic for legacy captures;
            # the missing physical dimensions remain a release-blocking error.
            height, bbox_center = 0., np.zeros(3)
        physical = qualify_episode(timestamps_s=t["timestamp"], positions_xyz=t["position_world"],
            yaw_rad=yaw, speeds_mps=t["speed"], lateral_acceleration_mps2=lateral,
            steering_fractions=t["steer"], reference=reference,
            vehicle_length_m=dimensions["length"], vehicle_width_m=dimensions["width"],
            vehicle_height_m=height, vehicle_bbox_center_xyz=bbox_center,
            rotation_world=t["rotation_world"], velocity_world=t["velocity_world"],
            collisions=len(manifest["collision_events"]) if "collision_events" in manifest else None,
            rendered_boundary_agreement=None, completed_maneuver=t["route_s"][-1]-t["route_s"][0] >= target,
            intended_progress_m=target, recovery=bool(manifest.get("recovery", False)),recovery_onset_s=manifest.get('recovery_onset_s',0.)).to_dict()
        if manifest.get('recovery_onset_s',0.):
            event=manifest.get('recovery_event',{});onset=event.get('onset_index')
            if type(onset) is not int or not 0<=onset<len(t['timestamp']):
                errors.append('measured_recovery_onset_missing')
            elif (abs(reference.project(t['position_world'][onset,:2])[1])<.15
                  or not np.isclose(t['timestamp'][onset]-t['timestamp'][0],manifest['recovery_onset_s'],atol=1e-6)):
                errors.append('measured_recovery_initial_offset_or_timing_invalid')
        # The pixel ledger proves marking visibility. Physical/rendered boundary
        # agreement is a separate reviewer-bound measurement, never supported>=4.
        review_path = (_member(root,manifest['automatic_review_file']) if manifest.get('automatic_review_file')
                       else root / "reviews" / (str(manifest["geometry_group_id"]) + ".json"))
        review = _json(review_path)
        if review.get("geometry_sha256") != sha256_file(shard / "geometry.json"):
            errors.append("review_geometry_binding_mismatch")
        if manifest.get("annotation_status")=="proposal_requires_rendered_review":
            for field in ("maneuver_phase_ranges","intent_required_route_range_m","desire_route_range_m"):
                if review.get(field)!=manifest.get(field):
                    errors.append("review_maneuver_annotation_mismatch:"+field)
        clip = _binding(review.get("clip"), root)
        reviewed_indices = set(review.get("label_indices", []))
        required_indices = {record["label_index"] for record in manifest["label_records"]
                            if record["visibility_category"] == "history_crop_loss"}
        if (review.get("decision") != "pass" or not review.get("reviewer") or
            set(review.get("phases", [])) != set(PHASES) or not required_indices.issubset(reviewed_indices)):
            errors.append("per_geometry_visual_review_incomplete")
        if review.get('review_mode')=='automated_measured_v1':
            from .automatic_review import review_errors
            errors.extend(review_errors(review,manifest,t,manifest['label_records']))
        boundary = review.get("rendered_boundary_measurement", {})
        if "rendered-boundary-report.json" in manifest.get("files",{}):
            boundary=rendered_boundary_status(shard,manifest,t)
            errors.extend(boundary["errors"])
        if (boundary.get("frame_count") != len(t["timestamp"]) or not np.isfinite(boundary.get("maximum_disagreement_m", np.nan))
            or boundary.get("maximum_disagreement_m", 1) > .05 or not boundary.get("method")):
            errors.append("rendered_boundary_agreement_measurement_missing_or_failed")
        else:
            physical["reasons"] = [reason for reason in physical["reasons"] if reason != "rendered_boundary_agreement_missing_or_failed"]
            physical["accepted"] = not physical["reasons"]
        if not physical["accepted"]:
            errors.extend("physical:" + reason for reason in physical["reasons"])
        qualification["bindings"]["review"] = {"path": str(review_path), "sha256": sha256_file(review_path)}
        qualification["bindings"]["review_clip"] = {"path": str(clip), "sha256": sha256_file(clip)}
    except (KeyError, ValueError, TypeError, OSError) as error:
        errors.append("physical_or_review_evidence_invalid:" + str(error))
    report = dict(schema_version=1, passed=not errors, errors=sorted(set(errors)), physical=physical,
        capture_envelope=envelope,
        label_count=built["label_count"], lateral_label_count=built["lateral_label_count"],
        bindings=qualification["bindings"], artifact_hashes={key: value for key, value in manifest["files"].items()
                                                          if key != "admission-report.json"},
        status="qualified_for_atomic_publication" if not errors else "diagnostic_not_admitted")
    write_json(shard / "admission-report.json", report)
    manifest["files"]["admission-report.json"] = sha256_file(shard / "admission-report.json")
    manifest["admission"] = report
    # Keep development status and accepted=false records until the publisher
    # rechecks bindings and performs its atomic transaction.
    write_json(shard / "manifest.json", manifest)
    return report
