"""Speed-independent spatial plan loss with a fixed original-model progress anchor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

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

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("loss settings must be finite and nonnegative")
        if (not 0 < self.sample_spacing_m <= .25 or self.lateral_scale_m <= 0
                or self.progress_scale_m <= 0 or self.clearance_scale_m <= 0
                or self.progress_weight <= 0 or self.lateral_weight <= 0):
            raise ValueError("positive geometry and progress terms are required")


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
    diagnostics = dict(valid_labels=0, path_samples=0, footprint_samples=0,
        supervised_distance_m=0., lateral_loss=0., progress_loss=0., heading_loss=0.,
        clearance_loss=0., mean_abs_lateral_m=0., max_clearance_violation_m=0.)
    for row, corridor in enumerate(corridors):
        idx, fraction, query, weight = _samples(reference[row, :, :2], corridor, cfg.sample_spacing_m)
        if not len(query):
            continue
        center, tangent, left, right, valid = corridor.sample(query)
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
            + cfg.heading_weight*heading_loss + cfg.clearance_weight*clearance_loss)
        diagnostics["valid_labels"] += 1
        diagnostics["path_samples"] += len(query)
        diagnostics["footprint_samples"] += int(body_valid.sum())
        diagnostics["supervised_distance_m"] += float(weight.sum())
        for name, value in (("lateral_loss", lateral_loss), ("progress_loss", progress_loss),
                            ("heading_loss", heading_loss), ("clearance_loss", clearance_loss),
                            ("mean_abs_lateral_m", average(lateral.abs()))):
            diagnostics[name] += float(value.detach())
        if body_valid.any():
            diagnostics["max_clearance_violation_m"] = max(diagnostics["max_clearance_violation_m"],
                float(violation[mask].detach().max()))
    count = diagnostics["valid_labels"]
    for key in ("lateral_loss", "progress_loss", "heading_loss", "clearance_loss", "mean_abs_lateral_m"):
        diagnostics[key] /= max(1, count)
    return total/max(1, count), diagnostics
