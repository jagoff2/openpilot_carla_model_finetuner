"""Privileged simulator geometry for a physical demonstration teacher only.

This module deliberately does not create ObservableReference, ObservedRoad,
ObservedObstacle, BehaviorTarget or EvidenceFrame objects. Simulator road edges
and actor footprints may plan a demonstrator, but they are never evidence that
the student could see those surfaces. Observed label reconstruction remains a
separate operation, using the actual camera history.

The bounded geometric search shares pure geometry checks with target.py. Its
result has a distinct type and an explicit prohibition on supervision use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from itertools import product
import json

import numpy as np

from lane_dataset.geometry import LaneReference
from .search import _clearance_offsets
from .evidence import _polygon
from .passing import PlannedPassingSpace,permission_issue
from .target import (BehaviorConfig, EgoFootprint, Excursion, _check_path,
    _geometry, _behavior_geometry, _project_vertices, _simple_polygon)


VERSION = "privileged_static_scene_teacher_geometry_v2"
SOURCE_SCOPE = "privileged_simulator_geometry_teacher_and_physical_qa_only"


@dataclass(frozen=True)
class PlannedObstacle:
    """Simulator actor envelope, not an observed footprint or visual proof."""
    identity: str
    polygon_xy: object
    kind: str
    motion: str = "unknown"

    def polygon(self):
        return _polygon(self.polygon_xy).copy()


def _digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _source_document(reference, obstacles, ego, config, source_metadata, recovery_m,
                     obstacle_tracking_reserve_m=0.,passing_space=None):
    document = dict(version=VERSION, source_scope=SOURCE_SCOPE, source_metadata=source_metadata,
        coordinate_frame="carla_world_xy_positive_y_right",
        reference=dict(left_xyz=reference.left_xyz.tolist(), right_xyz=reference.right_xyz.tolist(),
            s=reference.s.tolist(), left_spline=reference._left.c.tolist(),
            right_spline=reference._right.c.tolist(), center_spline=reference._center.c.tolist()),
        obstacles=[dict(identity=o.identity, polygon_xy=o.polygon().tolist(), kind=o.kind,
            motion=o.motion) for o in obstacles], ego=asdict(ego), config=asdict(config),
        minimum_retained_nominal_recovery_m=float(recovery_m))
    # Legacy zero-reserve captures retain their exact source shape and hash.
    if obstacle_tracking_reserve_m:
        document["obstacle_tracking_reserve_m"] = float(obstacle_tracking_reserve_m)
    if passing_space is not None:document['passing_space']=passing_space.document()
    return document


def _solution_document(s, excursions, feasible, reasons):
    return dict(station=s.tolist(), excursions=[asdict(e) for e in excursions],
        feasible=feasible, reasons=list(reasons))


@dataclass(frozen=True)
class PlannedTarget:
    """Map/actor-based teacher path. Intentionally not a BehaviorTarget subtype."""
    reference: LaneReference
    obstacles: tuple[PlannedObstacle, ...]
    ego: EgoFootprint
    config: BehaviorConfig
    s: np.ndarray
    excursions: tuple[Excursion, ...] = ()
    feasible: bool = False
    reasons: tuple[str, ...] = ()
    obstacle_ids: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)
    source_metadata: dict = field(default_factory=dict)
    minimum_retained_nominal_recovery_m: float = 5.
    contract_sha256: str = ""
    version: str = VERSION
    source_scope: str = SOURCE_SCOPE
    supervision_allowed: bool = False
    obstacle_tracking_reserve_m: float = 0.
    passing_space: PlannedPassingSpace | None = None

    def verify_contract(self):
        if (self.supervision_allowed is not False or self.version != VERSION
                or self.source_scope != SOURCE_SCOPE
                or tuple(o.identity for o in self.obstacles) != self.obstacle_ids):
            raise ValueError("planned target provenance contract changed")
        source = _source_document(self.reference, self.obstacles, self.ego, self.config,
            self.source_metadata, self.minimum_retained_nominal_recovery_m, self.obstacle_tracking_reserve_m,self.passing_space)
        solution = _solution_document(self.s, self.excursions, self.feasible, self.reasons)
        if self.contract_sha256 != _digest(dict(source=source, solution=solution)):
            raise ValueError("planned scene or teacher path changed after geometric qualification")

    def sample(self, query):
        """Teacher XY, tangent, permitted left/right edge and planned-support mask.

        This mask denotes planned geometry availability. It is never visibility
        or a training supervision mask. Invalid positions remain NaN.
        """
        self.verify_contract()
        q = np.asarray(query, float)
        finite = np.isfinite(q)
        valid = finite & (q >= self.s[0]) & (q <= self.s[-1]) & self.feasible
        safe = np.clip(np.where(finite, q, self.s[0]), self.s[0], self.s[-1])
        values = _behavior_geometry(self.reference, safe, self.excursions,self.passing_space)
        return (*tuple(np.where(valid[..., None], value, np.nan) for value in values), valid)

    @property
    def xy(self):
        return self.sample(self.s)[0]

    @property
    def nominal_xy(self):
        self.verify_contract()
        return _geometry(self.reference, self.s)[0]

    def profile(self):
        """Read back planned path geometry, never measured physical telemetry."""
        xy, tangent, divider, right, valid = self.sample(self.s)
        if not valid.all():
            return None
        left3, right3 = self.reference._left(self.s), self.reference._right(self.s)
        cross = right3[:, :2]-left3[:, :2]
        fraction = ((xy-left3[:, :2])*cross).sum(1)/np.maximum((cross*cross).sum(1), 1e-12)
        xyz = np.c_[xy, left3[:, 2]+fraction*(right3[:, 2]-left3[:, 2])]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        if np.any(np.diff(arc) <= 1e-8):
            raise ValueError("planned path has degenerate progress")
        yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        order = 2 if len(arc) > 2 else 1
        curvature = np.gradient(yaw, arc, edge_order=order)
        profile = dict(road_station_m=self.s.tolist(), path_arc_m=arc.tolist(),
            position_xyz=xyz.tolist(), tangent_xy=tangent.tolist(), yaw_rad=yaw.tolist(),
            curvature_1pm=curvature.tolist(), nominal_xy=self.nominal_xy.tolist(),
            virtual_divider_xy=((left3[:, :2]+right3[:, :2])*.5).tolist(), right_road_edge_xy=right.tolist(),
            scope="planned_geometry_not_physical_trajectory_or_observed_label")
        if self.passing_space is not None:
            profile['permitted_left_boundary_xy']=divider.tolist()
        return profile

    def document(self):
        """Serializable geometry, solver result and explicit teacher provenance."""
        self.verify_contract()
        return dict(version=self.version, source_scope=self.source_scope,
            supervision_allowed=False, training_ready=False, physical_qa_passed=False,
            contract_sha256=self.contract_sha256,
            source=_source_document(self.reference, self.obstacles, self.ego, self.config,
                self.source_metadata, self.minimum_retained_nominal_recovery_m, self.obstacle_tracking_reserve_m,self.passing_space),
            solution=_solution_document(self.s, self.excursions, self.feasible, self.reasons),
            diagnostics=self.diagnostics, planned_profile=self.profile())


def _nominal_distance(reference, first, last, spacing):
    if last <= first:
        return 0.
    q = np.linspace(first, last, max(2, int(np.ceil((last-first)/spacing))+1))
    xy = _geometry(reference, q)[0]
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())



def planned_scene_geometry(reference, obstacles=(), ego=None, *, config=None,
                           source_metadata=None, minimum_retained_nominal_recovery_m=5.,
                           obstacle_tracking_reserve_m=0.,passing_space=None):
    """Plan physical teacher geometry without manufacturing observed evidence.

    ``reference`` contains the full two-way pavement's outer edges, not a
    single lane's boundaries. The teacher remains inside its right half unless
    explicit source-bound static passing permission expands the event corridor. Every
    declared obstacle is a simulator QA envelope. Dynamic/unknown actors,
    insufficient body clearance and incomplete return/recovery yield no path.
    The physical controller must still enforce calibration and measured QA.
    A nonnegative obstacle tracking reserve expands only the planner's search
    clearance. Online body checks and actual rollout QA retain config limits.
    """
    if not isinstance(reference, LaneReference):
        raise TypeError("teacher planning requires explicit simulator LaneReference geometry")
    obstacles = tuple(obstacles)
    if any(not isinstance(o, PlannedObstacle) for o in obstacles):
        raise TypeError("teacher planning requires PlannedObstacle simulator envelopes")
    if passing_space is not None and not isinstance(passing_space,PlannedPassingSpace):
        raise TypeError('teacher passing requires privileged PlannedPassingSpace provenance')
    cfg, ego = config or BehaviorConfig(), ego or EgoFootprint()
    if not np.isfinite(minimum_retained_nominal_recovery_m) or minimum_retained_nominal_recovery_m < 5.:
        raise ValueError("teacher avoidance requires at least five metres nominal recovery")
    if (isinstance(obstacle_tracking_reserve_m, (bool, np.bool_))
            or not isinstance(obstacle_tracking_reserve_m, (int, float, np.number))
            or not np.isfinite(obstacle_tracking_reserve_m) or obstacle_tracking_reserve_m < 0.):
        raise ValueError("obstacle tracking reserve must be finite and nonnegative")
    obstacle_tracking_reserve_m = float(obstacle_tracking_reserve_m)
    metadata = {} if source_metadata is None else source_metadata
    if not isinstance(metadata, dict):
        raise TypeError("source metadata must be an explicit dictionary")
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    ref = reference
    corners = ego.corners()
    radius = float(np.linalg.norm(corners, axis=1).max())
    grid = np.linspace(ref.s[0], ref.s[-1], int(np.ceil((ref.s[-1]-ref.s[0])/cfg.sample_spacing_m))+1)
    derivative = .25*ref._left(grid, 1)[..., :2]+.75*ref._right(grid, 1)[..., :2]
    minimum_progress = float(np.linalg.norm(derivative, axis=1).min())
    guard = (radius+cfg.edge_clearance_m)/max(minimum_progress, 1e-6)
    first, last = float(ref.s[0]+guard), float(ref.s[-1]-guard)
    s = np.linspace(first, max(first, last),
        max(2, int(np.ceil(max(0., last-first)/cfg.sample_spacing_m))+1))
    s.flags.writeable = False
    ids = tuple(o.identity for o in obstacles)

    def result(reasons=(), excursions=(), diagnostics=None):
        reasons, excursions = tuple(reasons), tuple(excursions)
        source = _source_document(ref, obstacles, ego, cfg, metadata,
            minimum_retained_nominal_recovery_m, obstacle_tracking_reserve_m,passing_space)
        solution = _solution_document(s, excursions, not reasons, reasons)
        details = dict(diagnostics or {})
        if obstacle_tracking_reserve_m:
            details["obstacle_tracking_reserve_m"] = obstacle_tracking_reserve_m
            details["obstacle_reserve_scope"] = "planning_only_actual_controller_and_physical_QA_keep_config_clearance"
            surplus = details.get("minimum_obstacle_clearance_surplus_m")
            if surplus is not None:
                details["minimum_actual_obstacle_clearance_surplus_m"] = float(surplus+obstacle_tracking_reserve_m)
        return PlannedTarget(reference=ref, obstacles=obstacles, ego=ego, config=cfg,
            s=s, excursions=excursions, feasible=not reasons, reasons=reasons,
            obstacle_ids=ids, diagnostics=details, source_metadata=metadata,
            minimum_retained_nominal_recovery_m=float(minimum_retained_nominal_recovery_m),
            obstacle_tracking_reserve_m=obstacle_tracking_reserve_m,
            passing_space=passing_space,
            contract_sha256=_digest(dict(source=source, solution=solution)))

    if minimum_progress <= 1e-6:
        return result(("degenerate_planned_right_half_center",))
    if last <= first:
        return result(("insufficient_planned_body_support",))
    if len(set(ids)) != len(ids) or any(not isinstance(name, str) or not name for name in ids):
        return result(("planned_obstacle_identity_missing_or_duplicate",))
    issue=permission_issue(passing_space,ref,obstacles,observed=False)
    if issue:return result((issue,))
    qualified = []
    for obstacle in obstacles:
        if obstacle.motion != "static":
            return result(("planned_obstacle_motion_unknown_or_dynamic:"+obstacle.identity,))
        margin = cfg.obstacle_clearance(obstacle.kind)
        if margin is None:
            return result(("unsupported_planned_obstacle_class:"+obstacle.identity,))
        qualified.append((obstacle, obstacle.polygon(), margin+obstacle_tracking_reserve_m))
    _, _, divider, right = _geometry(ref, grid)
    ribbon = np.concatenate((divider, right[::-1]))
    if not _simple_polygon(ribbon):
        return result(("planned_right_half_ribbon_folds",))
    safe, reason, details = _check_path(ref, s, (), corners, ribbon, qualified, cfg)
    if safe:
        return result(diagnostics=details)
    if not reason.startswith("insufficient_obstacle_body_clearance:"):
        return result((reason,))
    half_width = float(np.abs(corners[:, 1]).max())
    half_length = float(np.abs(corners[:, 0]).max())
    groups = []
    for obstacle, polygon, margin in qualified:
        station, lateral = _project_vertices(ref, grid, polygon)
        if lateral.min() > half_width+margin or lateral.max() < -half_width-margin:
            continue
        groups.append(dict(lo=float(station.min()-half_length-margin),
            hi=float(station.max()+half_length+margin), low=float(lateral.min()-half_width-margin),
            high=float(lateral.max()+half_width+margin), ids=[obstacle.identity]))
    groups.sort(key=lambda item:item["lo"])
    if not groups:
        return result((reason,))
    merged = []
    for group in groups:
        if merged and group["lo"]-merged[-1]["hi"] < 2*cfg.min_transition_m+minimum_retained_nominal_recovery_m:
            previous = merged[-1]
            previous.update(hi=max(previous["hi"], group["hi"]), low=min(previous["low"], group["low"]),
                high=max(previous["high"], group["high"]), ids=previous["ids"]+group["ids"])
        else:
            merged.append(dict(group))
    alternatives = []
    # Search at most the half-lane's lateral span, allowing an actor-origin
    # offset within its measured bounding box. The exact whole-body check,
    # rather than this finite proposal bound, decides every candidate's safety.
    search_span = float(np.linalg.norm(ref._right(grid)[..., :2]-ref._left(grid)[..., :2], axis=1).max()/4
                        + np.linalg.norm(ego.center_xy))
    if passing_space is not None:search_span+=passing_space.opposing_width_m
    for group in merged:
        choices = []
        for offset in _clearance_offsets(group["low"], group["high"], search_span, cfg.maximum_alternatives):
            length = max(cfg.min_transition_m,
                np.sqrt(5.774*abs(offset)/cfg.maximum_added_second_derivative_inv_m),
                np.cbrt(60*abs(offset)/cfg.maximum_added_third_derivative_inv_m2))
            start, end = group["lo"]-length, group["hi"]+length
            if (start < s[0] or end > s[-1]
                    or _nominal_distance(ref, end, s[-1], cfg.sample_spacing_m)
                       < minimum_retained_nominal_recovery_m):
                continue
            choices.append(Excursion(start, group["lo"], group["hi"], end, float(offset), tuple(group["ids"])))
        if not choices:
            return result(("insufficient_planned_departure_return_and_recovery:"+",".join(group["ids"]),))
        alternatives.append(choices)
    attempted, failures = 0, set()
    for excursions in product(*alternatives):
        attempted += 1
        if attempted > cfg.maximum_alternatives:
            return result(("planned_avoidance_search_budget_exhausted",),
                diagnostics=dict(attempted_alternatives=attempted-1))
        if any(a.end_s > b.start_s or _nominal_distance(ref, a.end_s, b.start_s, cfg.sample_spacing_m)
               < minimum_retained_nominal_recovery_m for a, b in zip(excursions, excursions[1:])):
            failures.add("planned_avoidance_events_lack_separate_nominal_recovery")
            continue
        safe, reason, details = _check_path(ref, s, excursions, corners, ribbon, qualified, cfg,passing_space)
        if safe:
            details.update(attempted_alternatives=attempted,
                check_scope="planned_target_chords_swept_body_rotation_guard_not_physical_capture",
                static_scene_only=True, offset_refinement_step_m=.05,
                minimum_retained_nominal_recovery_m=float(minimum_retained_nominal_recovery_m))
            return result(excursions=excursions, diagnostics=details)
        failures.add(reason)
    return result(("no_planned_right_half_pass_with_complete_recovery", *sorted(failures)),
        diagnostics=dict(attempted_alternatives=attempted))
