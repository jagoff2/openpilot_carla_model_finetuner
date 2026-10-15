"""Speed-independent spatial plan loss with a fixed original-model progress anchor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

OBJECTIVE_VERSION = "observed_centerline_origin_recovery_v2"

@dataclass(frozen=True)
class LossConfig:
    sample_spacing_m: float = .25
    clearance_m: float = .15
    lateral_scale_m: float = .10
    progress_scale_m: float = 1.0
    clearance_scale_m: float = .10
    lateral_weight: float = 1.0
    progress_weight: float = 1.0
    heading_weight: float = .25
    clearance_weight: float = 1.0
    origin_weight: float = 1.0
    recovery_min_length_m: float = 2.0
    recovery_max_length_m: float = 20.0
    recovery_second_derivative_limit_inv_m: float = .06
    recovery_third_derivative_limit_inv_m2: float = .10
    recovery_min_forward_derivative: float = .5
    centered_position_tolerance_m: float = .001
    aligned_heading_tolerance_rad: float = .001

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("loss settings must be finite and nonnegative")
        if (not 0 < self.sample_spacing_m <= .25 or self.lateral_scale_m <= 0
                or self.progress_scale_m <= 0 or self.clearance_scale_m <= 0
                or self.progress_weight <= 0 or self.lateral_weight <= 0
                or self.origin_weight <= 0 or self.recovery_min_length_m <= 0
                or self.recovery_max_length_m < self.recovery_min_length_m
                or self.recovery_second_derivative_limit_inv_m <= 0
                or self.recovery_third_derivative_limit_inv_m2 <= 0
                or not 0 < self.recovery_min_forward_derivative <= 1):
            raise ValueError("positive geometry and progress terms are required")


def _correction(distance, length, offset, direction_delta):
    """Quintic Hermite displacement, with zero displacement/slope/curvature at L."""
    u = np.clip(np.asarray(distance, float)/length, 0, 1)
    h0 = 1-10*u**3+15*u**4-6*u**5
    h1 = u-6*u**3+8*u**4-3*u**5
    dh0 = (-30*u**2+60*u**3-30*u**4)/length
    dh1 = 1-18*u**2+32*u**3-15*u**4
    ddh0 = (-60*u+180*u**2-120*u**3)/length**2
    ddh1 = (-36*u+96*u**2-60*u**3)/length
    dddh0 = (-60+360*u-360*u**2)/length**3
    dddh1 = (-36+192*u-180*u**2)/length**2
    active = (np.asarray(distance) <= length)[..., None]
    values = (h0[..., None]*offset+length*h1[..., None]*direction_delta,
              dh0[..., None]*offset+dh1[..., None]*direction_delta,
              ddh0[..., None]*offset+ddh1[..., None]*direction_delta,
              dddh0[..., None]*offset+dddh1[..., None]*direction_delta)
    return tuple(value*active for value in values)


def _derivative_maxima(length, offset, direction_delta):
    """Exact polynomial extrema of correction derivative norms over [0,L]."""
    from numpy.polynomial import polynomial as poly
    maxima = []
    for a, b in ((np.array([0., -60., 180., -120.])/length**2,
                  np.array([0., -36., 96., -60.])/length),
                 (np.array([-60., 360., -360.])/length**3,
                  np.array([-36., 192., -180.])/length**2)):
        coefficients = a[:, None]*offset+b[:, None]*direction_delta
        square = np.zeros(1)
        for c in coefficients.T:
            square = poly.polyadd(square, poly.polymul(c, c))
        roots = poly.polyroots(poly.polyder(square))
        real = roots.real[(np.abs(roots.imag) < 1e-7) & (roots.real > 0) & (roots.real < 1)]
        values = np.stack([poly.polyval(np.r_[0., real, 1.], c) for c in coefficients.T], axis=1)
        maxima.append(float(np.linalg.norm(values, axis=1).max()))
    return maxima


@dataclass(frozen=True)
class RecoveryTarget:
    corridor: object
    length: float
    offset: np.ndarray
    direction_delta: np.ndarray

    def sample(self, query):
        center, tangent, left, right, valid = self.corridor.sample(query)
        if self.length:
            delta, slope, _, _ = _correction(np.asarray(query)-self.corridor.ego_s,
                self.length, self.offset, self.direction_delta)
            center = center+delta
            tangent = tangent+slope
            tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
        return center, tangent, left, right, valid


def recovery_target(corridor, config=None):
    """Fixed observed target from ego origin/tangent, then exact lane center.

    Limits bound derivatives of the added spatial correction, not vehicle
    acceleration or jerk. They are independent of student, speed and commands.
    Unobserved gaps, folded targets and footprint violations reject recovery.
    """
    cfg = config or LossConfig()
    center, tangent, _, _, valid = corridor.sample(np.array([corridor.ego_s]))
    if not valid[0]:
        return None
    offset = -center[0]
    direction_delta = np.array([1., 0.])-tangent[0]
    heading = abs(np.arctan2(tangent[0, 1], tangent[0, 0]))
    if (np.linalg.norm(offset) <= cfg.centered_position_tolerance_m
            and heading <= cfg.aligned_heading_tolerance_rad):
        return RecoveryTarget(corridor, 0., offset, direction_delta)
    idx = np.searchsorted(corridor.s, corridor.ego_s, side="right")-1
    idx = int(np.clip(idx, 0, len(corridor.valid)-1))
    first_gap = np.flatnonzero(~corridor.valid[idx:])
    end = corridor.s[idx+first_gap[0]] if len(first_gap) else corridor.s[-1]
    available = min(cfg.recovery_max_length_m, float(end-corridor.ego_s))
    if available < cfg.recovery_min_length_m:
        return None
    lengths = np.r_[np.arange(cfg.recovery_min_length_m, available, cfg.sample_spacing_m), available]
    footprint = _perimeter(corridor, cfg.sample_spacing_m)
    for length in lengths:
        distance = np.linspace(0, length, int(np.ceil(length/cfg.sample_spacing_m))+1)
        second_max, third_max = _derivative_maxima(length, offset, direction_delta)
        if (second_max > cfg.recovery_second_derivative_limit_inv_m
                or third_max > cfg.recovery_third_derivative_limit_inv_m2):
            continue
        _, slope, _, _ = _correction(distance, length, offset, direction_delta)
        q = corridor.ego_s+distance
        base_center, base_tangent, _, _, supported = corridor.sample(q)
        raw_tangent = base_tangent+slope
        if (not supported.all() or np.min((raw_tangent*base_tangent).sum(1))
                < cfg.recovery_min_forward_derivative):
            continue
        target = RecoveryTarget(corridor, float(length), offset, direction_delta)
        point, direction, _, _, _ = target.sample(q)
        normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
        body = point[:, None]+direction[:, None]*footprint[None, :, :1]+normal[:, None]*footprint[None, :, 1:]
        bc, bt, bl, br, body_valid = corridor.sample(q[:, None]+footprint[None, :, 0])
        bn = np.stack([-bt[..., 1], bt[..., 0]], axis=-1)
        body_offset = ((body-bc)*bn).sum(-1)
        low = ((bl-bc)*bn).sum(-1)+cfg.clearance_m
        high = ((br-bc)*bn).sum(-1)-cfg.clearance_m
        if np.any(body_valid & ((body_offset < low-1e-6) | (body_offset > high+1e-6))):
            continue
        return target
    return None


def _samples(reference_xy, corridor, spacing):
    # Distance comes from the frozen original checkpoint, not the recording and
    # not the trainable student's current predictions. No measured speed input.
    distance = np.linalg.norm(np.diff(reference_xy, axis=0), axis=1)
    if not np.isfinite(distance).all() or distance.sum() > 1000:
        raise ValueError("invalid original-model forward progress")
    station = corridor.ego_s + np.r_[0., np.cumsum(distance)]
    indices, fractions, queries, weights = [], [], [], []
    for i, ds in enumerate(distance):
        if ds <= 1e-6:
            continue
        count = max(2, int(np.ceil(ds/spacing)))
        # Include endpoints with trapezoid weights and intermediate path samples.
        f = np.linspace(0, 1, count+1)
        w = np.full(count+1, ds/count)
        w[[0, -1]] *= .5
        indices.extend([i]*len(f)); fractions.extend(f)
        queries.extend(station[i]+f*ds); weights.extend(w)
    return tuple(np.asarray(value) for value in (indices, fractions, queries, weights))


def _perimeter(corridor, spacing):
    x = np.linspace(-corridor.length/2, corridor.length/2, int(np.ceil(corridor.length/spacing))+1)
    y = np.linspace(-corridor.width/2, corridor.width/2, int(np.ceil(corridor.width/spacing))+1)
    points = np.concatenate([np.c_[x, np.full_like(x, sign*corridor.width/2)] for sign in (-1, 1)]
        + [np.c_[np.full_like(y, sign*corridor.length/2), y] for sign in (-1, 1)])
    c, s = np.cos(corridor.bbox_yaw), np.sin(corridor.bbox_yaw)
    return points @ np.array([[c, s], [-s, c]]) + np.asarray(corridor.bbox_center)


def centerline_loss(predictions, reference_predictions, corridors, *, predicted_yaw=None, config=None):
    """Return (loss, diagnostics) for canonical FRD XY paths [labels,points,2].

    Point count is model-independent. Callers explicitly decode their chosen
    model's outputs into XY metres and optionally yaw radians [labels,points].

    reference_predictions MUST come from a fixed original checkpoint using its
    own recurrent history. Detachment here protects its gradients; it does not
    turn a changing student into a fixed teacher. Reference chord lengths set
    route progress and immutable support masks. XY errors are split into lateral
    placement and tangent displacement, so stopping/shortening cannot erase loss.

    The path is piecewise linear between output positions. Heading and a sampled
    vehicle perimeter are scored throughout each supported segment. This is a
    differentiable sampled objective, not a continuous containment certificate.
    Geometry outside observed support is never extrapolated or supervised.
    """
    cfg = config or LossConfig()
    if (predictions.ndim != 3 or predictions.shape[-1] != 2 or predictions.shape[1] < 2
            or not predictions.is_floating_point() or not reference_predictions.is_floating_point()):
        raise ValueError("expected floating XY paths [labels,points,2] with at least two points")
    plan = predictions.float()
    reference = reference_predictions.detach().float().cpu().numpy()
    if len(plan) != len(corridors) or reference.shape != tuple(plan.shape):
        raise ValueError("prediction, reference and corridor rows must match")
    if predicted_yaw is not None:
        if (predicted_yaw.shape != predictions.shape[:2] or not predicted_yaw.is_floating_point()
                or predicted_yaw.device != predictions.device):
            raise ValueError("yaw must match path rows, points and device")
        predicted_yaw = predicted_yaw.float()
    if not np.isfinite(reference[:, :, :2]).all():
        raise ValueError("original-model path must be finite")
    # An empty supported batch still has a finite differentiable zero.
    zero = torch.nan_to_num(plan[:, :, :2], nan=0., posinf=0., neginf=0.).sum()*0
    total = zero
    diagnostics = dict(objective_version=OBJECTIVE_VERSION, valid_labels=0, path_samples=0, footprint_samples=0,
        unsupported_recovery_labels=0, recovery_labels=0, recovery_length_max_m=0., origin_loss=0.,
        supervised_distance_m=0., lateral_loss=0., progress_loss=0., heading_loss=0.,
        clearance_loss=0., mean_abs_lateral_m=0., max_clearance_violation_m=0.)
    for row, corridor in enumerate(corridors):
        idx, fraction, query, weight = _samples(reference[row, :, :2], corridor, cfg.sample_spacing_m)
        if not len(query):
            continue
        target = recovery_target(corridor, cfg)
        if target is None:
            diagnostics['unsupported_recovery_labels'] += 1
            continue
        center, tangent, left, right, valid = target.sample(query)
        if not valid.any():
            continue
        idx, fraction, query, weight = (x[valid] for x in (idx, fraction, query, weight))
        center, tangent = center[valid], tangent[valid]
        tensor = lambda value: torch.as_tensor(value, device=plan.device, dtype=plan.dtype)
        index = torch.as_tensor(idx, device=plan.device, dtype=torch.long)
        a, b = plan[row, index], plan[row, index+1]
        if not bool(torch.isfinite(torch.cat([a, b])).all()):
            raise ValueError("nonfinite supported student position")
        f, w = tensor(fraction)[:, None], tensor(weight)
        point = a[:, :2]*(1-f)+b[:, :2]*f
        t = tensor(tangent)
        normal = torch.stack([-t[:, 1], t[:, 0]], dim=1)
        error = point-tensor(center)
        lateral = (error*normal).sum(1)
        progress = (error*t).sum(1)
        average = lambda value: (value*w).sum()/w.sum()
        huber = lambda value: F.smooth_l1_loss(value, torch.zeros_like(value), reduction="none")
        if not bool(torch.isfinite(plan[row, 0]).all()):
            raise ValueError("nonfinite supported student origin")
        origin_loss = huber(plan[row, 0]/cfg.lateral_scale_m).sum()
        if predicted_yaw is not None:
            if not bool(torch.isfinite(predicted_yaw[row, 0])):
                raise ValueError("nonfinite supported student origin heading")
            origin_loss = origin_loss+1-torch.cos(predicted_yaw[row, 0])
        lateral_loss = average(huber(lateral/cfg.lateral_scale_m))
        progress_loss = average(huber(progress/cfg.progress_scale_m))
        direction = b[:, :2]-a[:, :2]
        direction = direction/torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-4)
        # At a degenerate student segment the target orientation keeps the body
        # rectangle nondegenerate. Progress loss still rejects stationary collapse.
        moving = torch.linalg.vector_norm(b[:, :2]-a[:, :2], dim=1, keepdim=True) > 1e-4
        body_direction = torch.where(moving, direction, t)
        body_normal = torch.stack([-body_direction[:, 1], body_direction[:, 0]], dim=1)
        heading_error = 1-(direction*t).sum(1)
        if predicted_yaw is not None:
            ya, yb = predicted_yaw[row, index], predicted_yaw[row, index+1]
            if not bool(torch.isfinite(torch.cat([ya, yb])).all()):
                raise ValueError("nonfinite supported student heading")
            yaw_delta = torch.atan2(torch.sin(yb-ya), torch.cos(yb-ya))
            yaw = ya+f[:, 0]*yaw_delta
            yaw_direction = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
            heading_error = heading_error + 1-(yaw_direction*t).sum(1)
            body_direction = yaw_direction
            body_normal = torch.stack([-body_direction[:, 1], body_direction[:, 0]], dim=1)
        heading_loss = average(heading_error)
        footprint = _perimeter(corridor, cfg.sample_spacing_m)
        # Fixed correspondence for every body sample. A student cannot remove a
        # penalty by moving its body into an unobserved area or changing a mask.
        body_q = query[:, None]+footprint[None, :, 0]
        bc, bt, bl, br, body_valid = corridor.sample(body_q)
        bn = np.stack([-bt[..., 1], bt[..., 0]], axis=-1)
        local = tensor(footprint)
        body = point[:, None, :]+body_direction[:, None, :]*local[None, :, :1]+body_normal[:, None, :]*local[None, :, 1:]
        offset = ((body-tensor(bc))*tensor(bn)).sum(-1)
        left_limit = tensor(((bl-bc)*bn).sum(-1))+cfg.clearance_m
        right_limit = tensor(((br-bc)*bn).sum(-1))-cfg.clearance_m
        violation = torch.relu(left_limit-offset)+torch.relu(offset-right_limit)
        mask = torch.as_tensor(body_valid, device=plan.device)
        body_weights = w[:, None]*mask
        clearance_loss = (huber(violation/cfg.clearance_scale_m)*body_weights).sum()/body_weights.sum().clamp_min(1e-9)
        total = total + (cfg.lateral_weight*lateral_loss + cfg.progress_weight*progress_loss
            + cfg.heading_weight*heading_loss + cfg.clearance_weight*clearance_loss
            + cfg.origin_weight*origin_loss)
        diagnostics["valid_labels"] += 1
        diagnostics['recovery_labels'] += int(target.length > 0)
        diagnostics['recovery_length_max_m'] = max(diagnostics['recovery_length_max_m'], target.length)
        diagnostics["path_samples"] += len(query)
        diagnostics["footprint_samples"] += int(body_valid.sum())
        diagnostics["supervised_distance_m"] += float(weight.sum())
        for name, value in (("lateral_loss", lateral_loss), ("progress_loss", progress_loss), ("origin_loss", origin_loss),
                            ("heading_loss", heading_loss), ("clearance_loss", clearance_loss),
                            ("mean_abs_lateral_m", average(lateral.abs()))):
            diagnostics[name] += float(value.detach())
        if body_valid.any():
            diagnostics["max_clearance_violation_m"] = max(diagnostics["max_clearance_violation_m"],
                float(violation[mask].detach().max()))
    count = diagnostics["valid_labels"]
    for key in ("lateral_loss", "progress_loss", "heading_loss", "clearance_loss", "origin_loss", "mean_abs_lateral_m"):
        diagnostics[key] /= max(1, count)
    return total/max(1, count), diagnostics
