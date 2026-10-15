"""Differentiable lateral supervision for qualified unmarked-road behavior.

This extension is deliberately outside the source-bound running v2 trainer.
Its explicit adapter can be used by any differentiable plan export. It does
not make an offline scenario or a rendered probe into an admitted episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch.nn import functional as F

from lane_training.centerline import LossConfig, _samples, centerline_loss, recovery_target
from .target import (BehaviorTarget, ObservedObstacle, _polygon_distance,
    _boundary_distance, _inside, _geometry, _offset, contract_fingerprint)
from .window import WindowTarget
from .passing import permitted_ribbon

OBJECTIVE_VERSION = "observed_unmarked_right_half_static_avoidance_future_v4"


@dataclass(frozen=True)
class BehaviorCorridor:
    """Ego-relative FRD target, including actual right-side road constraints.

    The caller transforms the observed road and obstacle geometry to the same
    current-ego coordinates before building the BehaviorTarget. Full-road
    center and virtual divider are never confused with the driving target.

    ``source_ego_s`` is the original observed full-road station supplied by the
    record. Exposed ``s`` and ``ego_s`` are actual behavior-path arc meters. The
    arc origin keeps the source support's first station value, preserving the
    original station convention on a straight, undisturbed path.
    """
    target: BehaviorTarget | WindowTarget
    source_ego_s: float | None
    obstacles: tuple[ObservedObstacle, ...] = ()
    _metric_s: np.ndarray = field(init=False, repr=False, compare=False)
    _source_s: np.ndarray = field(init=False, repr=False, compare=False)
    _ego_metric_s: float = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.target, (BehaviorTarget, WindowTarget)):
            raise TypeError("training requires an observed BehaviorTarget or WindowTarget")
        source = np.asarray(self.target.s, float)
        if (source.ndim != 1 or len(source) < 2 or not np.isfinite(source).all()
                or np.any(np.diff(source) <= 0)):
            raise ValueError("behavior support must be finite and strictly increasing")
        future_only = isinstance(self.target, WindowTarget) and self.target.source_ego_s is None
        if future_only and self.source_ego_s is not None:
            raise ValueError("future geometry has no observed ego station")
        if not future_only and (self.source_ego_s is None or not np.isfinite(self.source_ego_s)
                or not source[0] <= self.source_ego_s <= source[-1]):
            raise ValueError("ego station must lie inside behavior support")
        if (len({o.identity for o in self.obstacles}) != len(self.obstacles)
                or set(self.target.obstacle_ids) != {o.identity for o in self.obstacles}):
            raise ValueError("loss obstacles must match target evidence")
        self.verify_contract()
        # Fixed quadrature/inverse lookup depends only on the qualified observed
        # target. Neither student geometry nor recorded speed can alter masks or
        # correspondence. Include every behavior join and the ego knot exactly.
        count = max(2, int(np.ceil((source[-1]-source[0])/.05))+1)
        joins = [q for e in self.target.excursions for q in
            (e.start_s, e.plateau_start_s, e.plateau_end_s, e.end_s) if source[0] <= q <= source[-1]]
        observed = self.target.road.reference.s
        observed = observed[(observed >= source[0]) & (observed <= source[-1])]
        source = np.unique(np.r_[source, np.linspace(source[0], source[-1], count),
            [] if future_only else [self.source_ego_s], observed, joins])
        # Distinct grid constructors can express the same knot one ULP apart.
        # Those zero-length quadrature intervals are not geometric degeneracy.
        end = source[-1]
        source = source[np.r_[True,np.diff(source)>1e-9]]
        if end-source[-1] <= 1e-9:
            source[-1] = end
        nodes, weights = np.polynomial.legendre.leggauss(8)
        half = np.diff(source)/2
        query = (source[:-1]+half)[:, None]+half[:, None]*nodes[None]
        speed = _behavior_path_speed(self.target, query)
        interval_length = half*(speed*weights).sum(1)
        if not np.isfinite(interval_length).all() or np.any(interval_length <= 1e-10):
            raise ValueError("behavior target has degenerate metric progress")
        metric = source[0]+np.r_[0., np.cumsum(interval_length)]
        ego_metric = None if future_only else float(np.interp(self.source_ego_s, source, metric))
        source.flags.writeable = False
        metric.flags.writeable = False
        object.__setattr__(self, '_source_s', source)
        object.__setattr__(self, '_metric_s', metric)
        object.__setattr__(self, '_ego_metric_s', ego_metric)

    def verify_contract(self):
        target = self.target
        if isinstance(target, WindowTarget):
            target.verify_contract(self.obstacles)
            if ((self.source_ego_s is None) != (target.source_ego_s is None)
                    or (self.source_ego_s is not None and abs(self.source_ego_s-target.source_ego_s)>1e-6)):
                raise ValueError("loss ego station differs from observed future window")
            return
        if target.contract_sha256 is None:
            raise ValueError("behavior target has no bound scene contract")
        current = contract_fingerprint(target.road, self.obstacles, target.ego, target.config,target.passing_space)
        if current != target.contract_sha256:
            raise ValueError("behavior inputs do not match target scene contract")

    @property
    def s(self):
        return self._metric_s

    @property
    def ego_s(self):
        return self._ego_metric_s

    @property
    def source_stations(self):
        return self._source_s

    @property
    def valid(self):
        return np.full(len(self.s)-1, self.target.moving_supervision, dtype=bool)

    @property
    def length(self):
        return self.target.ego.length_m

    @property
    def width(self):
        return self.target.ego.width_m

    @property
    def bbox_center(self):
        return self.target.ego.center_xy

    @property
    def bbox_yaw(self):
        return self.target.ego.yaw_rad

    def sample(self, query):
        q = np.asarray(query, float)
        finite = np.isfinite(q)
        supported = finite & (q >= self.s[0]) & (q <= self.s[-1])
        safe = np.clip(np.where(finite, q, self.s[0]), self.s[0], self.s[-1])
        source = np.interp(safe.ravel(), self.s, self.source_stations).reshape(q.shape)
        *values, valid = self.target.sample(source)
        valid = valid & supported
        # The common loss uses false masks as zero weights. Keep unsupported
        # placeholders finite so NaN*0 cannot poison an otherwise supported row.
        return (*tuple(np.where(valid[..., None], value, 0.) for value in values), valid)


def _behavior_path_speed(target, query):
    """Analytic d(behavior XY)/d(observed-road station), including detours."""
    ref = target.road.reference
    first = .25*ref._left(query, 1)[..., :2]+.75*ref._right(query, 1)[..., :2]
    second = .25*ref._left(query, 2)[..., :2]+.75*ref._right(query, 2)[..., :2]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    tangent = first/np.maximum(norm, 1e-12)
    normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    tangent_derivative = (second-tangent*(second*tangent).sum(-1, keepdims=True))/np.maximum(norm, 1e-12)
    normal_derivative = np.stack((-tangent_derivative[..., 1], tangent_derivative[..., 0]), axis=-1)
    offset, slope = _offset(query, target.excursions)
    derivative = first+slope[..., None]*normal+offset[..., None]*normal_derivative
    return np.linalg.norm(derivative, axis=-1)


def _sat_clearance(body, obstacle):
    """Conservative signed convex-polygon separation, differentiable in body.

    Positive values are a lower bound on Euclidean separation; negative values
    indicate overlap. Including body and obstacle edge normals also detects
    obstacles lying entirely inside the car, which point-only losses miss.
    """
    edges = torch.cat((torch.roll(body, -1, 1)-body,
        (torch.roll(obstacle, -1, 0)-obstacle)[None].expand(len(body), -1, -1)), dim=1)
    axes = torch.stack((-edges[..., 1], edges[..., 0]), dim=-1)
    axes = axes/torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(1e-8)
    bp = torch.einsum("nvd,nad->nav", body, axes)
    op = torch.einsum("vd,nad->nav", obstacle, axes)
    gaps = torch.maximum(op.amin(-1)-bp.amax(-1), bp.amin(-1)-op.amax(-1))
    return gaps.amax(-1)


def _body_points(center, tangent, corners):
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)
    return center[:, None]+tangent[:, None]*corners[None, :, :1]+normal[:, None]*corners[None, :, 1:]


def _qualified_origin_recovery(corridor, cfg):
    """Do not let the origin correction erase an obstacle-avoidance constraint."""
    recovery = recovery_target(corridor, cfg)
    if recovery is None:
        return False
    end = min(corridor.s[-1], corridor.ego_s+max(recovery.length, cfg.sample_spacing_m))
    q = np.linspace(corridor.ego_s, end, int(np.ceil((end-corridor.ego_s)/cfg.sample_spacing_m))+1)
    xy, tangent, _, _, valid = recovery.sample(q)
    if not valid.all():
        return False
    corners = corridor.target.ego.corners()
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    body = xy[:, None]+tangent[:, None]*corners[None, :, :1]+normal[:, None]*corners[None, :, 1:]
    from scipy.spatial import ConvexHull
    radius = float(np.linalg.norm(corners, axis=1).max())
    ref = corridor.target.road.reference
    grid = np.linspace(ref.s[0], ref.s[-1], int(np.ceil((ref.s[-1]-ref.s[0])/cfg.sample_spacing_m))+1)
    _, _, divider, edge = _geometry(ref, grid)
    ribbon = np.concatenate((divider, edge[::-1]))
    if corridor.target.passing_space is not None:
        ribbon=permitted_ribbon(ref,cfg.sample_spacing_m,corridor.target.excursions,corridor.target.passing_space)
    required_road_margin = max(cfg.clearance_m, corridor.target.config.edge_clearance_m)
    for i in range(len(body)-1):
        cloud = np.concatenate((body[i], body[i+1]))
        hull = cloud[ConvexHull(cloud).vertices]
        angle = np.arccos(np.clip(tangent[i]@tangent[i+1], -1., 1.))
        guard = radius*angle*angle/8
        if (not _inside(hull, ribbon).all()
                or _boundary_distance(hull, ribbon) < required_road_margin+guard-1e-8):
            return False
        for obstacle in corridor.obstacles:
            margin = corridor.target.config.obstacle_clearance(obstacle.kind)
            if _polygon_distance(hull, obstacle.polygon()) < margin+guard:
                return False
    return True


def behavior_loss(predictions, reference_predictions, corridors, *, predicted_yaw=None, config=None):
    """Score ego-relative XY plans against right-side/avoidance/recovery targets.

    Fixed original-checkpoint progress determines correspondence and masks.
    Geometry and actor envelopes are labels only, never additional model inputs.
    Unknown/moving obstacles or infeasible origin recovery mask moving labels;
    no stopped-car or collision path is substituted as a lateral demonstration.
    """
    cfg = config or LossConfig()
    if (predictions.shape != reference_predictions.shape or predictions.ndim != 3
            or predictions.shape[-1] != 2 or predictions.shape[1] < 2
            or not predictions.is_floating_point() or not reference_predictions.is_floating_point()):
        raise ValueError("student and original plans must have equal [B,N,2] shapes")
    if len(corridors) != len(predictions):
        raise ValueError("one behavior corridor per plan required")
    if predicted_yaw is not None and (predicted_yaw.shape != predictions.shape[:2]
            or not predicted_yaw.is_floating_point() or predicted_yaw.device != predictions.device):
        raise ValueError("yaw must match path rows, points and device")
    for corridor in corridors:
        corridor.verify_contract()
    cfg = replace(cfg, clearance_m=max([cfg.clearance_m]+[c.target.config.edge_clearance_m for c in corridors]))
    future = [i for i, c in enumerate(corridors) if c.source_ego_s is None]
    if future:
        from .future_loss import visible_future_loss
        selected = torch.as_tensor(future, device=predictions.device)
        future_value, report = visible_future_loss(predictions[selected],reference_predictions[selected],
            [corridors[i] for i in future],predicted_yaw=None if predicted_yaw is None else predicted_yaw[selected],config=cfg)
        others = [i for i in range(len(corridors)) if i not in future]
        if others:
            ix = torch.as_tensor(others,device=predictions.device)
            value, extra = behavior_loss(predictions[ix],reference_predictions[ix],[corridors[i] for i in others],
                predicted_yaw=None if predicted_yaw is None else predicted_yaw[ix],config=cfg)
            n, m = report['valid_labels'], extra['valid_labels']
            future_value = (future_value*n+value*m)/max(1,n+m)
            report['legacy_window_diagnostics'] = extra
            report['valid_labels'] = n+m
        report.update(objective_version=OBJECTIVE_VERSION,masked_labels=len(corridors)-report['valid_labels'])
        return future_value,report
    usable = [i for i, c in enumerate(corridors)
        if c.target.moving_supervision and _qualified_origin_recovery(c, cfg)]
    if not usable:
        return torch.nan_to_num(predictions, nan=0., posinf=0., neginf=0.).sum()*0, dict(
            objective_version=OBJECTIVE_VERSION, valid_labels=0, masked_labels=len(corridors), obstacle_loss=0.)
    indices = torch.as_tensor(usable, device=predictions.device)
    selected = [corridors[i] for i in usable]
    loss, report = centerline_loss(predictions[indices], reference_predictions[indices], selected,
        predicted_yaw=None if predicted_yaw is None else predicted_yaw[indices], config=cfg)
    obstacle_total = loss*0
    obstacle_rows = 0
    reference = reference_predictions.detach().float().cpu().numpy()
    for row in usable:
        corridor = corridors[row]
        if not corridor.obstacles:
            continue
        idx, fraction, query, weight = _samples(reference[row], corridor, cfg.sample_spacing_m)
        if not len(query):
            continue
        _, tangent, _, _, valid = corridor.sample(query)
        if not valid.any():
            continue
        idx, fraction, weight, tangent = [x[valid] for x in (idx, fraction, weight, tangent)]
        tensor = lambda x: torch.as_tensor(x, device=predictions.device, dtype=torch.float32)
        ix = torch.as_tensor(idx, device=predictions.device, dtype=torch.long)
        a, b = predictions[row, ix].float(), predictions[row, ix+1].float()
        f = tensor(fraction)[:, None]
        point = (1-f)*a+f*b
        delta = b-a
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        direction = torch.where(norm > 1e-4, delta/norm.clamp_min(1e-4), tensor(tangent))
        if predicted_yaw is not None:
            ya, yb = predicted_yaw[row, ix], predicted_yaw[row, ix+1]
            yaw = ya+f[:, 0]*torch.atan2(torch.sin(yb-ya), torch.cos(yb-ya))
            direction = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        body = _body_points(point, direction, tensor(corridor.target.ego.corners()))
        w = tensor(weight)
        penalty = loss*0
        for obstacle in corridor.obstacles:
            clearance = _sat_clearance(body, tensor(obstacle.polygon()))
            margin = corridor.target.config.obstacle_clearance(obstacle.kind)
            violation = torch.relu(margin-clearance)/cfg.clearance_scale_m
            penalty = penalty+(F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none")*w).sum()/w.sum()
        obstacle_total = obstacle_total+penalty
        obstacle_rows += 1
    obstacle_total = obstacle_total/max(1, len(usable))
    loss = loss+cfg.clearance_weight*obstacle_total
    report.update(objective_version=OBJECTIVE_VERSION, masked_labels=len(corridors)-len(usable),
        obstacle_rows=obstacle_rows, obstacle_loss=float(obstacle_total.detach()),
        obstacle_constraint="sampled_convex_body_separation_not_a_dynamic_safety_certificate")
    return loss, report


def model_behavior_loss(predictions, reference_predictions, corridors, *, adapter, config=None):
    """Use the chosen model's explicit plan decoder, including CD210's decoder."""
    prediction = adapter(predictions)
    reference = adapter(reference_predictions.detach())
    return behavior_loss(prediction.xy, reference.xy, corridors, predicted_yaw=prediction.yaw, config=config)
