"""Causal future windows of observed unmarked-road behavior.

The road and actors must already be independently observed. This module never
queries a map, makes evidence objects, or certifies completion of an avoidance
event. Excursion anchors may precede or follow the visible window; only geometry
inside its present, body-supported future interval is evaluated and exposed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from itertools import product
import json

import numpy as np

from lane_dataset.geometry import EvidenceFrame, ObservableReference
from .search import _clearance_offsets
from .passing import ObservedPassingSpace,permission_issue
from .target import (BehaviorConfig, EgoFootprint, Excursion, ObservedObstacle,
    ObservedRoad, _check_path, _evidence_issue, _geometry, _behavior_geometry, _project_vertices,
    _simple_polygon, contract_fingerprint)


VERSION = "observed_right_half_future_window_v2"
SCOPE = "causal_observed_future_window_not_complete_event_qualification"


def _fingerprint(road, obstacles, ego, config, source_ego_s, s, excursions,
                 moving, reasons, evidence, obstacle_ids, diagnostics,passing_space=None):
    document = dict(version=VERSION, scope=SCOPE,
        scene_sha256=contract_fingerprint(road, obstacles, ego, config,passing_space),
        source_ego_s=None if source_ego_s is None else float(source_ego_s), supported_stations=np.asarray(s).tolist(),
        excursions=[asdict(event) for event in excursions], moving_supervision=bool(moving),
        reasons=list(reasons), evidence_frames=[asdict(e) for e in evidence],
        obstacle_ids=list(obstacle_ids), diagnostics=diagnostics)
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class WindowTarget:
    """Distinct observed future target; never a fabricated complete target."""
    road: ObservedRoad
    ego: EgoFootprint
    config: BehaviorConfig
    source_ego_s: float | None
    s: np.ndarray
    excursions: tuple[Excursion, ...] = ()
    moving_supervision: bool = False
    reasons: tuple[str, ...] = ()
    evidence_frames: tuple[EvidenceFrame, ...] = ()
    obstacle_ids: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)
    contract_sha256: str = ""
    version: str = VERSION
    scope: str = SCOPE
    complete_event_qualified: bool = False
    _obstacles: tuple[ObservedObstacle, ...] = field(default=(), repr=False, compare=False)
    passing_space: ObservedPassingSpace | None = None

    def verify_contract(self, obstacles=None):
        obstacles = self._obstacles if obstacles is None else tuple(obstacles)
        if (self.version != VERSION or self.scope != SCOPE
                or self.complete_event_qualified is not False):
            raise ValueError("observed window provenance contract changed")
        current = _fingerprint(self.road, obstacles, self.ego, self.config,
            self.source_ego_s, self.s, self.excursions, self.moving_supervision,
            self.reasons, self.evidence_frames, self.obstacle_ids, self.diagnostics,self.passing_space)
        if current != self.contract_sha256:
            raise ValueError("observed window inputs or qualified solution changed")

    def sample(self, query):
        self.verify_contract()
        q = np.asarray(query, float)
        finite = np.isfinite(q)
        valid = finite & (q >= self.s[0]) & (q <= self.s[-1]) & self.moving_supervision
        ref = self.road.reference
        safe = np.clip(np.where(finite, q, ref.s[0]), ref.s[0], ref.s[-1])
        values = _behavior_geometry(ref, safe, self.excursions,self.passing_space)
        return (*tuple(np.where(valid[..., None], value, np.nan) for value in values), valid)

    @property
    def xy(self):
        return self.sample(self.s)[0]

    @property
    def nominal_xy(self):
        self.verify_contract()
        return _geometry(self.road.reference,
            np.clip(self.s, self.road.reference.s[0], self.road.reference.s[-1]))[0]


def build_behavior_window(road, obstacles=(), ego=None, *, source_ego_s=None, config=None,passing_space=None):
    """Score the remaining observed path without requiring past maneuver pixels.

    ``source_ego_s`` is current ego progress projected on the independently
    observed nominal right-side center, before this target is constructed. It
    must not come from a teacher plan or a student prediction. The caller binds
    that projection to the actual pose in its replayed record.

    If source_ego_s is None, this is exclusively visible future geometry. No
    claim is made about the hidden road between the current ego and that span.
    Correspondence must be anchored by the frozen original model's path where
    it enters the observed span, never by projecting ego onto a clipped endpoint.

    Every supplied actor is validated, including actors behind the ego. Nothing
    is silently omitted: unknown/moving actors or endpoint-clipped projections
    mask supervision. A separate causal rear-clearance proof is needed before a
    caller omits an actor from its scene. Complete event traversal and recovery
    remain physical episode QA, never a property of this future window.
    """
    if not isinstance(road, ObservedRoad) or not isinstance(road.reference, ObservableReference):
        raise TypeError("window requires independently observed whole-road boundaries")
    obstacles = tuple(obstacles)
    if any(not isinstance(o, ObservedObstacle) for o in obstacles):
        raise TypeError("window requires explicitly observed obstacles")
    ref, cfg, ego = road.reference, config or BehaviorConfig(), ego or EgoFootprint()
    if source_ego_s is not None and (isinstance(source_ego_s, (bool, np.bool_)) or not np.isfinite(source_ego_s)
            or not ref.s[0] <= source_ego_s <= ref.s[-1]):
        raise ValueError("actual ego station must lie in observed road support")
    source_ego_s = None if source_ego_s is None else float(source_ego_s)
    corners = ego.corners()
    radius = float(np.linalg.norm(corners, axis=1).max())
    grid = np.linspace(ref.s[0], ref.s[-1],
        int(np.ceil((ref.s[-1]-ref.s[0])/cfg.sample_spacing_m))+1)
    derivative = .25*ref._left(grid, 1)[..., :2]+.75*ref._right(grid, 1)[..., :2]
    minimum_progress = float(np.linalg.norm(derivative, axis=1).min())
    guard = (radius+cfg.edge_clearance_m)/max(minimum_progress, 1e-6)
    available_start, available_end = float(ref.s[0]+guard), float(ref.s[-1]-guard)
    start = source_ego_s if source_ego_s is not None else available_start
    end = max(start, available_end)
    s = np.linspace(start, end,
        max(2, int(np.ceil((end-start)/cfg.sample_spacing_m))+1))
    s.flags.writeable = False
    evidence = tuple(dict.fromkeys(e for row in ref.evidence_frames for e in row))
    evidence = tuple(dict.fromkeys((*evidence, *(e for o in obstacles for e in o.evidence_frames))))
    ids = tuple(o.identity for o in obstacles)

    def result(reasons=(), excursions=(), diagnostics=None):
        reasons, excursions = tuple(reasons), tuple(excursions)
        details = dict(scope=SCOPE, source_ego_s=source_ego_s,
            origin_geometry_supported=source_ego_s is not None,
            complete_event_qualified=False, observed_road_span_m=[float(ref.s[0]), float(ref.s[-1])],
            body_supported_road_span_m=[available_start, available_end],
            future_query_span_m=[float(s[0]), float(s[-1])],
            past_departure_geometry_required=False, actors_automatically_omitted=[])
        details.update(diagnostics or {})
        digest = _fingerprint(road, obstacles, ego, cfg, source_ego_s, s, excursions,
            not reasons, reasons, evidence, ids, details,passing_space)
        return WindowTarget(road, ego, cfg, source_ego_s, s, excursions, not reasons,
            reasons, evidence, ids, details, digest, _obstacles=obstacles,passing_space=passing_space)

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
    if source_ego_s is not None and source_ego_s < available_start-1e-8:
        return result(("actual_ego_rear_body_not_supported_by_observed_road",))
    if available_end-start < cfg.sample_spacing_m:
        return result(("insufficient_observed_future_body_support",))
    if len(set(ids)) != len(ids) or any(not isinstance(name, str) or not name for name in ids):
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
    _, _, divider, right = _geometry(ref, grid)
    ribbon = np.concatenate((divider, right[::-1]))
    if not _simple_polygon(ribbon):
        return result(("observed_right_half_ribbon_folds",))

    groups = []
    half_width = float(np.abs(corners[:, 1]).max())
    half_length = float(np.abs(corners[:, 0]).max())
    behind_future_only = []
    for obstacle, polygon, margin in qualified:
        station, lateral = _project_vertices(ref, grid, polygon)
        # A clipped nearest projection cannot establish an actor's road order.
        # In particular it cannot prove that an unseen actor is safely behind.
        if np.any(station <= ref.s[0]+1e-6) or np.any(station >= ref.s[-1]-1e-6):
            if source_ego_s is None:
                # A still-observed actor may be behind this future-only span
                # while alongside ego. Do not claim ego has passed it. Establish
                # a separating plane for every nominal future body vertex and
                # retain the actor in all actual swept-polygon clearance tests.
                xy, tangent, *_ = _geometry(ref, s)
                normal = np.stack((-tangent[:, 1],tangent[:, 0]),axis=-1)
                body = xy[:,None]+tangent[:,None]*corners[None,:,:1]+normal[:,None]*corners[None,:,1:]
                axis = tangent[0]
                if (np.min((body-xy[0])@axis)-np.max((polygon-xy[0])@axis) > margin
                        and np.all((polygon-xy[0])@axis < 0)):
                    behind_future_only.append(obstacle.identity)
                    continue
            return result(("obstacle_station_not_supported_without_endpoint_clipping:"+obstacle.identity,))
        if lateral.min() > half_width+margin or lateral.max() < -half_width-margin:
            continue
        groups.append(dict(lo=float(station.min()-half_length-margin),
            hi=float(station.max()+half_length+margin), low=float(lateral.min()-half_width-margin),
            high=float(lateral.max()+half_width+margin), ids=[obstacle.identity]))
    groups.sort(key=lambda group: group["lo"])
    merged = []
    for group in groups:
        if merged and group["lo"]-merged[-1]["hi"] < 2*cfg.min_transition_m+(5. if passing_space is not None else 0.):
            previous = merged[-1]
            previous.update(hi=max(previous["hi"], group["hi"]), low=min(previous["low"], group["low"]),
                high=max(previous["high"], group["high"]), ids=previous["ids"]+group["ids"])
        else:
            merged.append(dict(group))
    search_span = float(np.linalg.norm(ref._right(grid)[..., :2]-ref._left(grid)[..., :2], axis=1).max()/4
                        + np.linalg.norm(ego.center_xy))
    if passing_space is not None:search_span+=passing_space.opposing_width_m
    alternatives = []
    for group in merged:
        choices = []
        for offset in _clearance_offsets(group["low"], group["high"], search_span, cfg.maximum_alternatives):
            length = max(cfg.min_transition_m,
                np.sqrt(5.774*abs(offset)/cfg.maximum_added_second_derivative_inv_m),
                np.cbrt(60*abs(offset)/cfg.maximum_added_third_derivative_inv_m2))
            # These are obstacle-relative phase anchors. No road geometry is
            # evaluated at an anchor outside observed support.
            choices.append(Excursion(group["lo"]-length, group["lo"], group["hi"],
                group["hi"]+length, float(offset), tuple(group["ids"])))
        alternatives.append(choices)
    attempted, failures = 0, set()
    for excursions in product(*alternatives):
        attempted += 1
        if attempted > cfg.maximum_alternatives:
            return result(("observed_future_avoidance_search_budget_exhausted",),
                diagnostics=dict(attempted_alternatives=attempted-1))
        if any(a.end_s > b.start_s for a, b in zip(excursions, excursions[1:])):
            failures.add("avoidance_transitions_overlap")
            continue
        safe, reason, details = _check_path(ref, s, excursions, corners, ribbon, qualified, cfg,passing_space)
        if safe:
            details.update(attempted_alternatives=attempted, static_scene_only=True,
                observed_actors_behind_future_span=behind_future_only,
                check_scope="observed_future_target_chords_swept_body_rotation_guard",
                complete_event_inside_query_span=all(e.start_s >= s[0] and e.end_s <= s[-1] for e in excursions))
            return result(excursions=excursions, diagnostics=details)
        failures.add(reason)
    return result(("no_body_qualified_observed_future_window", *sorted(failures)),
        diagnostics=dict(attempted_alternatives=attempted))
