"""Right-hand unmarked-road targets with bounded static-obstacle avoidance.

Inputs are independently observed *whole-road* edges and observed obstacle
footprints. No map coordinates, simulator actors or future observations are
queried here. Positive local/world Y is right for a +X traversal, as in CARLA
and comma FRD. A target is geometry for a static observed scene, not a dynamic
collision guarantee. Unknown/moving obstacles mask moving supervision.

The default contract yields when a complete pass and return cannot fit inside
the observed right half. Explicit replayable passing permission can expand only
a static parked-car avoidance event into an observed clear opposing strip.
Nominal driving and recovery remain centered in the right half. Yield does not
invent a longitudinal target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from lane_dataset.geometry import EvidenceFrame, ObservableReference
from .search import _clearance_offsets
from .passing import ObservedPassingSpace,permission_issue,permitted_edges,permitted_ribbon

VERSION = "observed_right_half_static_avoidance_v2"


@dataclass(frozen=True)
class ObservedRoad:
    reference: ObservableReference
    left_role: str = "road_edge"
    right_role: str = "road_edge"


@dataclass(frozen=True)
class EgoFootprint:
    length_m: float = 4.5
    width_m: float = 1.8
    center_xy: tuple[float, float] = (0., 0.)
    yaw_rad: float = 0.

    def corners(self):
        if (not np.isfinite([self.length_m, self.width_m, self.yaw_rad, *self.center_xy]).all()
                or min(self.length_m, self.width_m) <= 0):
            raise ValueError("invalid ego footprint")
        x, y = self.length_m/2, self.width_m/2
        p = np.array([[-x, -y], [x, -y], [x, y], [-x, y]])
        c, s = np.cos(self.yaw_rad), np.sin(self.yaw_rad)
        return p @ np.array([[c, s], [-s, c]]) + np.asarray(self.center_xy)


@dataclass(frozen=True)
class ObservedObstacle:
    identity: str
    polygon_xy: object
    kind: str
    evidence_frames: tuple[EvidenceFrame, ...]
    motion: str = "unknown"

    def polygon(self):
        points = np.asarray(self.polygon_xy, float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all()):
            raise ValueError("obstacle footprint must be a finite convex polygon")
        hull = ConvexHull(points)
        if len(hull.vertices) != len(points):
            raise ValueError("obstacle footprint must be convex with distinct vertices")
        # Canonical ordering avoids accepting self-crossing input polygons.
        if not (np.all(np.diff(np.r_[hull.vertices, hull.vertices[0]+len(points)]) % len(points) == 1)
                or np.all(np.diff(np.r_[hull.vertices, hull.vertices[0]-len(points)]) % len(points) == len(points)-1)):
            raise ValueError("obstacle polygon vertices must be in boundary order")
        return points[hull.vertices].copy()


@dataclass(frozen=True)
class BehaviorConfig:
    sample_spacing_m: float = .20
    maximum_observed_edge_gap_m: float = 8.
    maximum_evidence_age_s: float = 5.
    edge_clearance_m: float = .20
    parked_vehicle_clearance_m: float = .60
    pedestrian_clearance_m: float = 1.20
    static_object_clearance_m: float = .60
    min_transition_m: float = 6.
    maximum_added_second_derivative_inv_m: float = .06
    maximum_added_third_derivative_inv_m2: float = .10
    maximum_alternatives: int = 32

    def __post_init__(self):
        values = np.asarray(list(vars(self).values()), float)
        if (not np.isfinite(values).all() or np.any(values <= 0)
                or self.sample_spacing_m > .25 or self.maximum_alternatives < 1):
            raise ValueError("positive finite behavior settings and spacing <= .25 m required")

    def obstacle_clearance(self, kind):
        return {"parked_vehicle": self.parked_vehicle_clearance_m,
                "pedestrian": self.pedestrian_clearance_m,
                "static_object": self.static_object_clearance_m}.get(kind)


@dataclass(frozen=True)
class Excursion:
    start_s: float
    plateau_start_s: float
    plateau_end_s: float
    end_s: float
    offset_m: float
    obstacle_ids: tuple[str, ...]


def _smooth(u):
    u = np.clip(u, 0., 1.)
    return 10*u**3-15*u**4+6*u**5, 30*u**2-60*u**3+30*u**4


def _offset(q, excursions):
    value, slope = np.zeros_like(q), np.zeros_like(q)
    for event in excursions:
        approach = (q >= event.start_s) & (q < event.plateau_start_s)
        plateau = (q >= event.plateau_start_s) & (q <= event.plateau_end_s)
        recovery = (q > event.plateau_end_s) & (q <= event.end_s)
        length = event.plateau_start_s-event.start_s
        h, dh = _smooth((q-event.start_s)/length)
        value += event.offset_m*np.where(approach, h, plateau.astype(float))
        slope += event.offset_m*np.where(approach, dh/length, 0.)
        length = event.end_s-event.plateau_end_s
        h, dh = _smooth((q-event.plateau_end_s)/length)
        value += event.offset_m*np.where(recovery, 1-h, 0.)
        slope -= event.offset_m*np.where(recovery, dh/length, 0.)
    return value, slope


def _geometry(reference, q, excursions=()):
    left = reference._left(q)[..., :2]
    right = reference._right(q)[..., :2]
    nominal = .25*left+.75*right
    first = .25*reference._left(q, 1)[..., :2]+.75*reference._right(q, 1)[..., :2]
    second = .25*reference._left(q, 2)[..., :2]+.75*reference._right(q, 2)[..., :2]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    tangent = first/np.maximum(norm, 1e-12)
    normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
    dt = (second-tangent*(second*tangent).sum(-1, keepdims=True))/np.maximum(norm, 1e-12)
    dn = np.stack([-dt[..., 1], dt[..., 0]], axis=-1)
    offset, slope = _offset(q, excursions)
    xy = nominal+offset[..., None]*normal
    direction = first+slope[..., None]*normal+offset[..., None]*dn
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    return xy, direction, (left+right)*.5, right


def _behavior_geometry(reference,q,excursions=(),passing_space=None):
    xy,tangent,divider,right=_geometry(reference,q,excursions)
    if passing_space is not None:
        divider,right=permitted_edges(reference,q,excursions,passing_space)
    return xy,tangent,divider,right


def _nominal_length(reference,first,last,spacing):
    if last<=first:return 0.
    q=np.linspace(first,last,max(2,int(np.ceil((last-first)/spacing))+1))
    return float(np.linalg.norm(np.diff(_geometry(reference,q)[0],axis=0),axis=1).sum())


@dataclass(frozen=True)
class BehaviorTarget:
    road: ObservedRoad
    ego: EgoFootprint
    config: BehaviorConfig
    s: np.ndarray
    excursions: tuple[Excursion, ...] = ()
    moving_supervision: bool = False
    reasons: tuple[str, ...] = ()
    evidence_frames: tuple[EvidenceFrame, ...] = ()
    obstacle_ids: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)
    version: str = VERSION
    contract_sha256: str | None = None
    passing_space: ObservedPassingSpace | None = None

    def sample(self, query):
        """XY, unit tangent, virtual-left edge, right edge, observed-valid mask.

        Values outside body-qualified support are NaN and never extrapolated.
        Blocked targets return a false mask, including their diagnostic nominal
        geometry, so a caller cannot accidentally train through an obstruction.
        """
        q = np.asarray(query, float)
        finite = np.isfinite(q)
        valid = finite & (q >= self.s[0]) & (q <= self.s[-1]) & self.moving_supervision
        safe = np.clip(np.where(finite, q, self.s[0]), self.s[0], self.s[-1])
        values = _behavior_geometry(self.road.reference, safe, self.excursions,self.passing_space)
        return (*tuple(np.where(valid[..., None], value, np.nan) for value in values), valid)

    @property
    def xy(self):
        return self.sample(self.s)[0]

    @property
    def nominal_xy(self):
        return _geometry(self.road.reference, self.s)[0]


def contract_fingerprint(road, obstacles, ego, config,passing_space=None):
    """Bind the exact observed scene and geometry implementation cache.

    The immutable digest is checked against current inputs before each loss.
    IDs alone are never enough to substitute a different obstacle footprint.
    Spline coefficients are included so mutating an observed reference's cached
    geometry cannot evade the source-array binding.
    """
    ref = road.reference
    def frame(e):
        return {"frame_id": e.frame_id, "timestamp_s": float(e.timestamp_s), "camera": e.camera}
    document = dict(version=VERSION, road=dict(left_role=road.left_role, right_role=road.right_role,
        left=ref.left_xyz.tolist(), right=ref.right_xyz.tolist(), s=ref.s.tolist(),
        decision_time_s=float(ref.decision_time_s), evidence=[[frame(e) for e in row] for row in ref.evidence_frames],
        left_spline=ref._left.c.tolist(), right_spline=ref._right.c.tolist()),
        ego=asdict(ego), config=asdict(config), obstacles=[])
    for obstacle in sorted(obstacles, key=lambda o: o.identity):
        points = obstacle.polygon()
        first = int(np.lexsort((points[:, 1], points[:, 0]))[0])
        points = np.roll(points, -first, axis=0)
        document['obstacles'].append(dict(identity=obstacle.identity, polygon_xy=points.tolist(),
            kind=obstacle.kind, motion=obstacle.motion, evidence=[frame(e) for e in obstacle.evidence_frames]))
    if passing_space is not None:document['passing_space']=passing_space.document()
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _evidence_issue(frames, now, maximum_age):
    if (not frames or not np.isfinite(now) or any(not isinstance(e, EvidenceFrame)
            or not np.isfinite(e.timestamp_s) or e.timestamp_s > now+1e-8
            or e.camera not in ("narrow", "wide", "img", "big_img") for e in frames)):
        return "missing_or_future"
    if any(now-e.timestamp_s > maximum_age+1e-8 for e in frames):
        return "stale"
    return None


def _point_segment_distance(points, a, b):
    delta = b-a
    t = np.clip(((points-a)*delta).sum(-1)/np.maximum((delta*delta).sum(-1), 1e-18), 0., 1.)
    return np.linalg.norm(points-a-t[..., None]*delta, axis=-1)


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _polygon_distance(a, b):
    """Exact segment distance, including edge intersections and containment."""
    if _inside(a[:1], b)[0] or _inside(b[:1], a)[0]:
        return 0.
    a0, a1 = a[:, None], np.roll(a, -1, axis=0)[:, None]
    b0, b1 = b[None], np.roll(b, -1, axis=0)[None]
    aa, bb = a1-a0, b1-b0
    denominator = _cross(aa, bb)
    nonparallel = np.abs(denominator) > 1e-12
    denominator = np.where(nonparallel, denominator, 1.)
    t, u = _cross(b0-a0, bb)/denominator, _cross(b0-a0, aa)/denominator
    if np.any(nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
        return 0.
    return float(min(_point_segment_distance(a0, b0, b1).min(),
        _point_segment_distance(a1, b0, b1).min(),
        _point_segment_distance(b0, a0, a1).min(),
        _point_segment_distance(b1, a0, a1).min()))


def _boundary_distance(a, b):
    # Same segment-distance calculation without polygon-containment shortcut.
    a0, a1 = a[:, None], np.roll(a, -1, axis=0)[:, None]
    b0, b1 = b[None], np.roll(b, -1, axis=0)[None]
    aa, bb = a1-a0, b1-b0
    denominator = _cross(aa, bb)
    nonparallel = np.abs(denominator) > 1e-12
    denominator = np.where(nonparallel, denominator, 1.)
    t, u = _cross(b0-a0, bb)/denominator, _cross(b0-a0, aa)/denominator
    if np.any(nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
        return 0.
    return float(min(_point_segment_distance(a0, b0, b1).min(),
        _point_segment_distance(a1, b0, b1).min(),
        _point_segment_distance(b0, a0, a1).min(),
        _point_segment_distance(b1, a0, a1).min()))


def _inside(points, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    p = np.asarray(points)[:, None]
    crosses = (a[None, :, 1] > p[..., 1]) != (b[None, :, 1] > p[..., 1])
    dy = b[:, 1]-a[:, 1]
    x = a[None, :, 0]+(p[..., 1]-a[None, :, 1])*(b-a)[None, :, 0]/np.where(np.abs(dy) > 1e-14, dy, 1.)[None]
    return np.count_nonzero(crosses & (p[..., 0] < x), axis=1) % 2 == 1


def _simple_polygon(polygon):
    """Reject observed ribbon folds; adjacent segments may share endpoints."""
    n = len(polygon)
    for i in range(n-2):
        indices = np.arange(i+2, n)
        if i == 0:
            indices = indices[indices != n-1]
        a, delta = polygon[i], polygon[(i+1) % n]-polygon[i]
        b, other = polygon[indices], polygon[(indices+1) % n]-polygon[indices]
        denom = _cross(delta, other)
        good = np.abs(denom) > 1e-12
        denom = np.where(good, denom, 1.)
        t, u = _cross(b-a, other)/denom, _cross(b-a, delta)/denom
        if np.any(good & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
            return False
    return True


def _body(xy, tangent, corners):
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    return xy[:, None]+corners[None, :, :1]*tangent[:, None]+corners[None, :, 1:]*normal[:, None]


def _check_path(reference, s, excursions, corners, ribbon, obstacles, cfg,passing_space=None):
    if passing_space is not None:
        ribbon=permitted_ribbon(reference,cfg.sample_spacing_m,excursions,passing_space)
        if not _simple_polygon(ribbon):
            return False,'permitted_road_space_ribbon_folds',{}
    xy, tangent, *_ = _geometry(reference, s, excursions)
    body = _body(xy, tangent, corners)
    radius = np.linalg.norm(corners, axis=1).max()
    min_edge, min_obstacle = np.inf, np.inf
    for i in range(len(s)-1):
        points = np.concatenate([body[i], body[i+1]])
        swept = points[ConvexHull(points).vertices]
        # A rotating corner's circular arc lies within this distance of its
        # endpoint chord. Inflate both edge and obstacle clearances accordingly.
        angle = np.arccos(np.clip(tangent[i]@tangent[i+1], -1., 1.))
        rotation_guard = radius*(1-np.cos(angle/2))
        if not _inside(swept, ribbon).all():
            return False, "vehicle_footprint_leaves_right_half" if passing_space is None else 'vehicle_footprint_leaves_permitted_road_space', {}
        edge = _boundary_distance(swept, ribbon)-rotation_guard
        min_edge = min(min_edge, edge)
        if edge < cfg.edge_clearance_m-1e-8:
            return False, "insufficient_right_half_body_clearance" if passing_space is None else 'insufficient_permitted_road_body_clearance', {}
        for obstacle, polygon, margin in obstacles:
            distance = _polygon_distance(swept, polygon)-rotation_guard
            min_obstacle = min(min_obstacle, distance-margin)
            if distance < margin-1e-8:
                return False, "insufficient_obstacle_body_clearance:"+obstacle.identity, {}
    return True, "", dict(minimum_road_edge_clearance_m=float(min_edge),
        minimum_obstacle_clearance_surplus_m=None if np.isinf(min_obstacle) else float(min_obstacle))


def _project_vertices(reference, grid, vertices):
    nominal, tangent, *_ = _geometry(reference, grid)
    a, delta = nominal[:-1], np.diff(nominal, axis=0)
    t = np.clip(((vertices[:, None]-a[None])*delta[None]).sum(-1)/
        np.maximum((delta*delta).sum(-1), 1e-12), 0, 1)
    projected = a[None]+t[..., None]*delta[None]
    chosen = np.argmin(((vertices[:, None]-projected)**2).sum(-1), axis=1)
    row = np.arange(len(vertices))
    station = grid[chosen]+t[row, chosen]*np.diff(grid)[chosen]
    normal = np.stack([-tangent[chosen, 1], tangent[chosen, 0]], axis=1)
    lateral = ((vertices-projected[row, chosen])*normal).sum(-1)
    return station, lateral


def build_behavior_target(road: ObservedRoad, obstacles=(), ego=None, *, config=None,passing_space=None):
    """Create a body-qualified right-half path, or explicitly mask movement.

    Static means evidenced stationary for this bounded scene; callers must
    re-evaluate whenever evidence changes. Pedestrians receive a larger margin.
    Discrete path chords, body sweeps and obstacle polygons are checked together.
    Sampling bounds are explicit; this is not a continuous spline certificate.
    """
    cfg, ego = config or BehaviorConfig(), ego or EgoFootprint()
    if not isinstance(road, ObservedRoad) or not isinstance(road.reference, ObservableReference):
        raise TypeError("only independently observed full-road boundaries are accepted")
    ref = road.reference
    corners = ego.corners()
    radius = float(np.linalg.norm(corners, axis=1).max())
    road_grid = np.linspace(ref.s[0], ref.s[-1], int(np.ceil((ref.s[-1]-ref.s[0])/cfg.sample_spacing_m))+1)
    nominal_derivative = .25*ref._left(road_grid, 1)[..., :2]+.75*ref._right(road_grid, 1)[..., :2]
    minimum_progress = float(np.linalg.norm(nominal_derivative, axis=1).min())
    # Reference station is whole-road-midpoint arc. An offset curve has a
    # different metric length, so endpoint body support must account for that.
    guard = (radius+cfg.edge_clearance_m)/max(minimum_progress, 1e-6)
    available_start, available_end = float(ref.s[0]+guard), float(ref.s[-1]-guard)
    s = np.linspace(available_start, max(available_start, available_end),
        max(2, int(np.ceil(max(0., available_end-available_start)/cfg.sample_spacing_m))+1))
    evidence = tuple(dict.fromkeys(e for row in ref.evidence_frames for e in row))
    obstacles = tuple(obstacles)
    ids = tuple(o.identity for o in obstacles)
    def result(reasons=(), excursions=(), diagnostics=None):
        digest = None
        try:
            digest = contract_fingerprint(road, obstacles, ego, cfg,passing_space)
        except (ValueError, TypeError, AttributeError):
            if not reasons:
                raise
        return BehaviorTarget(road, ego, cfg, s, tuple(excursions), not reasons, tuple(reasons),
            evidence, ids, diagnostics or {}, contract_sha256=digest,passing_space=passing_space)
    if road.left_role != "road_edge" or road.right_role != "road_edge":
        return result(("full_road_edges_not_established",))
    if len(ref.evidence_frames) != len(ref.s):
        return result(("road_evidence_missing_or_future",))
    for row in ref.evidence_frames:
        issue = _evidence_issue(row, ref.decision_time_s, cfg.maximum_evidence_age_s)
        if issue:
            return result(("road_evidence_"+issue,))
    if minimum_progress <= 1e-6:
        return result(("degenerate_observed_right_half_center",))
    for edge in (ref.left_xyz, ref.right_xyz):
        if np.linalg.norm(np.diff(edge, axis=0), axis=1).max() > cfg.maximum_observed_edge_gap_m:
            return result(("observed_road_edge_gap_exceeds_limit",))
    if available_end <= available_start:
        return result(("insufficient_observed_body_support",))
    if len(set(ids)) != len(ids) or any(not name for name in ids):
        return result(("obstacle_identity_missing_or_duplicate",))
    issue=permission_issue(passing_space,ref,obstacles,observed=True,maximum_age_s=cfg.maximum_evidence_age_s)
    if issue:return result((issue,))
    if passing_space is not None:
        evidence=tuple(dict.fromkeys((*evidence,*passing_space.evidence_frames)))
    qualified = []
    for obstacle in obstacles:
        issue = _evidence_issue(obstacle.evidence_frames, ref.decision_time_s, cfg.maximum_evidence_age_s)
        if issue:
            return result(("obstacle_evidence_"+issue+":"+obstacle.identity,))
        if obstacle.motion != "static":
            return result(("obstacle_motion_unknown_or_dynamic:"+obstacle.identity,))
        margin = cfg.obstacle_clearance(obstacle.kind)
        if margin is None:
            return result(("unsupported_obstacle_class:"+obstacle.identity,))
        qualified.append((obstacle, obstacle.polygon(), margin))
        evidence = tuple(dict.fromkeys((*evidence, *obstacle.evidence_frames)))
    _, _, left, right = _geometry(ref, road_grid)
    ribbon = np.concatenate([left, right[::-1]])
    if not _simple_polygon(ribbon):
        return result(("observed_right_half_ribbon_folds",))
    safe, reason, details = _check_path(ref, s, (), corners, ribbon, qualified, cfg)
    if safe:
        return result(diagnostics=details)
    if not reason.startswith("insufficient_obstacle_body_clearance:"):
        return result((reason,))
    groups = []
    half_width = float(np.abs(corners[:, 1]).max())
    half_length = float(np.abs(corners[:, 0]).max())
    for obstacle, polygon, margin in qualified:
        station, lateral = _project_vertices(ref, road_grid, polygon)
        if lateral.min() > half_width+margin or lateral.max() < -half_width-margin:
            continue
        groups.append(dict(lo=float(station.min()-half_length-margin),
            hi=float(station.max()+half_length+margin),
            low=float(lateral.min()-half_width-margin),
            high=float(lateral.max()+half_width+margin), ids=[obstacle.identity]))
    groups.sort(key=lambda g: g['lo'])
    if not groups:
        return result((reason,))
    # Merge nearby obstacles so recovery never aims back into a second object.
    merged = []
    for group in groups:
        if merged and group['lo']-merged[-1]['hi'] < 2*cfg.min_transition_m+(5. if passing_space is not None else 0.):
            last = merged[-1]
            last.update(hi=max(last['hi'], group['hi']), low=min(last['low'], group['low']),
                high=max(last['high'], group['high']), ids=last['ids']+group['ids'])
        else:
            merged.append(dict(group))
    alternatives = []
    search_span = float(np.linalg.norm(ref._right(road_grid)[..., :2]-ref._left(road_grid)[..., :2], axis=1).max()/4
                        + np.linalg.norm(ego.center_xy))
    if passing_space is not None:search_span+=passing_space.opposing_width_m
    for group in merged:
        choices = []
        # A small extra margin avoids numerical/tessellation contact at an
        # analytically exact clearance. Both sides stay within the right half.
        for offset in _clearance_offsets(group['low'], group['high'], search_span, cfg.maximum_alternatives):
            length = max(cfg.min_transition_m,
                np.sqrt(5.774*abs(offset)/cfg.maximum_added_second_derivative_inv_m),
                np.cbrt(60*abs(offset)/cfg.maximum_added_third_derivative_inv_m2))
            start, end = group['lo']-length, group['hi']+length
            if start < s[0] or end > s[-1]:
                continue
            if passing_space is not None and _nominal_length(ref,end,s[-1],cfg.sample_spacing_m)<5.:
                continue
            choices.append(Excursion(start, group['lo'], group['hi'], end, float(offset), tuple(group['ids'])))
        if not choices:
            return result(("insufficient_observed_departure_and_return:"+','.join(group['ids']),))
        alternatives.append(choices)
    attempted, reasons = 0, set()
    for excursions in product(*alternatives):
        attempted += 1
        if attempted > cfg.maximum_alternatives:
            return result(("avoidance_search_budget_exhausted",), diagnostics={"attempted_alternatives": attempted-1})
        if any(a.end_s > b.start_s for a, b in zip(excursions, excursions[1:])):
            reasons.add("avoidance_transitions_overlap")
            continue
        if passing_space is not None and any(_nominal_length(ref,a.end_s,b.start_s,cfg.sample_spacing_m)<5.
                for a,b in zip(excursions,excursions[1:])):
            reasons.add('avoidance_events_lack_five_metre_nominal_recovery');continue
        safe, failure, details = _check_path(ref, s, excursions, corners, ribbon, qualified, cfg,passing_space)
        if safe:
            details.update(attempted_alternatives=attempted,
                check_scope="sampled_target_chords_with_swept_body_rotation_guard",
                static_scene_only=True)
            return result(excursions=excursions, diagnostics=details)
        reasons.add(failure)
    return result(("no_right_half_pass_with_complete_recovery", *sorted(reasons)),
        diagnostics={"attempted_alternatives": attempted})
