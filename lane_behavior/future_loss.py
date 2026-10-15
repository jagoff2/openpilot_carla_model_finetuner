"""Lateral loss on observed future geometry with no fabricated ego corridor.

The frozen original path fixes where its forward progress crosses an observed
target cross-section. Only that correspondence, not student placement, controls
support. The unseen prefix receives no road-geometry target. Origin anchoring
and supported tangent progress prevent translation and shortened-plan escapes.
"""
import numpy as np
import torch
from torch.nn import functional as F

from lane_training.centerline import _perimeter


def fixed_future_samples(reference, corridor, spacing):
    reference = np.asarray(reference, float)
    distance = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    if not np.isfinite(reference).all() or distance.sum() > 1000:
        raise ValueError("invalid frozen original path")
    arc = np.r_[0., np.cumsum(distance)]
    # Find the first supported target section crossed in the forward direction.
    # The road may start ahead of ego or extend behind it through causal history.
    # Multiple forward crossings would make route correspondence ambiguous.
    anchors = np.linspace(corridor.s[0], corridor.s[-1],
        max(2, int(np.ceil((corridor.s[-1]-corridor.s[0])/spacing))+1))
    center, tangent, _, _, valid = corridor.sample(anchors)
    anchor = None
    for q, point, direction, supported in zip(anchors, center, tangent, valid):
        if not supported:
            continue
        signed = (reference-point)@direction
        crossing = np.flatnonzero((signed[:-1] <= 0) & (signed[1:] > 0)
            & (np.diff(signed) > 1e-6) & (distance > 1e-6))
        if len(crossing) != 1:
            continue
        i = int(crossing[0])
        fraction = -signed[i]/(signed[i+1]-signed[i])
        anchor = (float(q), float(arc[i]+fraction*distance[i]))
        break
    if anchor is None:
        return (np.array([], int), *(np.array([]) for _ in range(3)))
    indices, fractions, queries, weights = [], [], [], []
    for i, ds in enumerate(distance):
        if ds <= 1e-6:
            continue
        count = max(2, int(np.ceil(ds/spacing)))
        f = np.linspace(0, 1, count+1)
        w = np.full(count+1, ds/count); w[[0, -1]] *= .5
        indices.extend([i]*len(f)); fractions.extend(f)
        queries.extend(anchor[0]+arc[i]+f*ds-anchor[1]); weights.extend(w)
    return tuple(np.asarray(v) for v in (indices, fractions, queries, weights))


def visible_future_loss(predictions, reference_predictions, corridors, *, predicted_yaw, config):
    from .loss import _sat_clearance, _body_points
    cfg, plan = config, predictions.float()
    references = reference_predictions.detach().float().cpu().numpy()
    total = torch.nan_to_num(plan, nan=0., posinf=0., neginf=0.).sum()*0
    names = ("lateral_loss", "progress_loss", "heading_loss", "clearance_loss", "origin_loss", "obstacle_loss")
    report = dict(valid_labels=0, path_samples=0, footprint_samples=0,
        supervised_distance_m=0., future_geometry_labels=0,
        correspondence="frozen_original_forward_cross_section_then_metric_arc",
        hidden_origin_road_supervision=False, **dict.fromkeys(names, 0.))
    for row, corridor in enumerate(corridors):
        if not corridor.target.moving_supervision:
            continue
        idx, fraction, query, weight = fixed_future_samples(references[row], corridor, cfg.sample_spacing_m)
        if not len(query):
            continue
        center, tangent, _, _, valid = corridor.sample(query)
        if not valid.any():
            continue
        idx, fraction, query, weight, center, tangent = [v[valid] for v in (idx, fraction, query, weight, center, tangent)]
        tensor = lambda value: torch.as_tensor(value, device=plan.device, dtype=plan.dtype)
        ix = torch.as_tensor(idx, device=plan.device, dtype=torch.long)
        a, b = plan[row, ix], plan[row, ix+1]
        if not torch.isfinite(torch.cat((a, b, plan[row, :1]))).all():
            raise ValueError("nonfinite supported student geometry")
        f, w, t = tensor(fraction)[:, None], tensor(weight), tensor(tangent)
        point = (1-f)*a+f*b
        normal = torch.stack((-t[:, 1], t[:, 0]), dim=-1)
        error = point-tensor(center)
        huber = lambda value: F.smooth_l1_loss(value, torch.zeros_like(value), reduction="none")
        average = lambda value: (value*w).sum()/w.sum()
        lateral = average(huber((error*normal).sum(-1)/cfg.lateral_scale_m))
        progress = average(huber((error*t).sum(-1)/cfg.progress_scale_m))
        norm = torch.linalg.vector_norm(b-a, dim=-1, keepdim=True)
        path_direction = (b-a)/norm.clamp_min(1e-4)
        direction = torch.where(norm > 1e-4, path_direction, t)
        heading_error = 1-(path_direction*t).sum(-1)
        origin = huber(plan[row, 0]/cfg.lateral_scale_m).sum()
        if predicted_yaw is not None:
            ya, yb = predicted_yaw[row, ix].float(), predicted_yaw[row, ix+1].float()
            if not torch.isfinite(torch.cat((ya, yb, predicted_yaw[row, :1]))).all():
                raise ValueError("nonfinite supported student heading")
            yaw = ya+f[:, 0]*torch.atan2(torch.sin(yb-ya), torch.cos(yb-ya))
            direction = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
            heading_error = heading_error+1-(direction*t).sum(-1)
            origin = origin+1-torch.cos(predicted_yaw[row, 0])
        heading = average(heading_error)
        perimeter = _perimeter(corridor, cfg.sample_spacing_m)
        bc, bt, bl, br, body_valid = corridor.sample(query[:, None]+perimeter[None, :, 0])
        bn = np.stack((-bt[..., 1], bt[..., 0]), axis=-1)
        body = _body_points(point, direction, tensor(perimeter))
        offset = ((body-tensor(bc))*tensor(bn)).sum(-1)
        violation = (torch.relu(tensor(((bl-bc)*bn).sum(-1))+cfg.clearance_m-offset)
            +torch.relu(offset-tensor(((br-bc)*bn).sum(-1))+cfg.clearance_m))
        mask = torch.as_tensor(body_valid, device=plan.device)
        bw = w[:, None]*mask
        clearance = (huber(violation/cfg.clearance_scale_m)*bw).sum()/bw.sum().clamp_min(1e-9)
        obstacle_loss = total*0
        corners = _body_points(point, direction, tensor(corridor.target.ego.corners()))
        for obstacle in corridor.obstacles:
            gap = _sat_clearance(corners, tensor(obstacle.polygon()))
            margin = corridor.target.config.obstacle_clearance(obstacle.kind)
            obstacle_loss = obstacle_loss+average(huber(torch.relu(margin-gap)/cfg.clearance_scale_m))
        total = total+cfg.lateral_weight*lateral+cfg.progress_weight*progress+cfg.heading_weight*heading+cfg.clearance_weight*(clearance+obstacle_loss)+cfg.origin_weight*origin
        report["valid_labels"] += 1; report["future_geometry_labels"] += 1
        report["path_samples"] += len(query); report["footprint_samples"] += int(body_valid.sum())
        report["supervised_distance_m"] += float(weight.sum())
        for key, value in zip(names, (lateral, progress, heading, clearance, origin, obstacle_loss)):
            report[key] += float(value.detach())
    count = max(1, report["valid_labels"])
    for key in names:
        report[key] /= count
    return total/count, report
