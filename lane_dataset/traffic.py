"""Optional physics-driven surrounding traffic and independent obstruction QA.

Only the capture owner ticks. Full lane geometry in this module is validation
metadata; no object here supplies an ego controller reference or student input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import queue
import random

import numpy as np

from .config import Config


class TrafficRejected(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__("traffic rejected: " + ", ".join(report["reasons"]))


@dataclass(frozen=True)
class TrafficConfig:
    count: int = 2
    speed_mps: float = 2.5
    corridor_clearance_m: float = .15
    body_clearance_m: float = .15
    spawn_ego_clearance_m: float = 8.
    spawn_traffic_clearance_m: float = 8.
    center_sample_step_m: float = .25
    candidate_step_m: float = 8.
    maximum_candidates: int = 100

    def __post_init__(self):
        if self.count not in (2, 3, 4):
            raise ValueError("surrounding traffic requires exactly 2, 3, or 4 vehicles")
        values = [self.speed_mps, self.corridor_clearance_m, self.body_clearance_m,
                  self.spawn_ego_clearance_m, self.spawn_traffic_clearance_m,
                  self.center_sample_step_m, self.candidate_step_m]
        if not all(math.isfinite(value) and value > 0 for value in values) or self.speed_mps > 10:
            raise ValueError("traffic distances and speed must be positive and finite; speed <=10 m/s")
        if self.maximum_candidates < self.count:
            raise ValueError("insufficient bounded spawn attempts")


def convex_hull(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("footprint requires finite XY points")
    ordered = sorted(set(map(tuple, points)))
    def half(items):
        result = []
        for point in items:
            while len(result) > 1:
                a, b = np.asarray(result[-2]), np.asarray(result[-1])
                c = np.asarray(point)
                if _cross(b-a, c-b) > 1e-12:
                    break
                result.pop()
            result.append(point)
        return result
    hull = np.asarray(half(ordered)[:-1] + half(ordered[::-1])[:-1])
    if len(hull) < 3:
        raise ValueError("degenerate body footprint")
    return hull


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _point_segment_distance(points, starts, ends):
    delta = ends-starts
    fraction = np.clip(np.sum((points-starts)*delta, axis=-1) /
                       np.maximum(np.sum(delta*delta, axis=-1), 1e-20), 0, 1)
    return np.linalg.norm(points-starts-fraction[..., None]*delta, axis=-1)


def polyline_polygon_distance(line, polygon):
    """Exact XY distance between straight segments and a convex body polygon."""
    line, polygon = np.asarray(line, float), np.asarray(polygon, float)
    starts, ends = line[:-1], line[1:]
    edges = np.roll(polygon, -1, axis=0)-polygon
    if np.any(np.all(_cross(edges[None], line[:, None]-polygon[None]) >= -1e-10, axis=1)):
        return 0.
    distance = math.inf
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        ab, cd = ends-starts, second-first
        # Proper crossings, with endpoint contacts handled by distances below.
        if np.any((_cross(ab, first-starts)*_cross(ab, second-starts) < 0) &
                  (_cross(cd, starts-first)*_cross(cd, ends-first) < 0)):
            return 0.
        distance = min(distance, float(np.min(_point_segment_distance(starts, first, second))),
                       float(np.min(_point_segment_distance(ends, first, second))),
                       float(np.min(_point_segment_distance(first, starts, ends))),
                       float(np.min(_point_segment_distance(second, starts, ends))))
    return distance


def polygon_distance(first, second):
    # Check containment in both directions as well as all edge intersections.
    return min(polyline_polygon_distance(np.vstack([first, first[0]]), second),
               polyline_polygon_distance(np.vstack([second, second[0]]), first))


@dataclass(frozen=True)
class BodyFootprint:
    actor_id: int
    vertices_xyz: np.ndarray
    rotation_matrix: np.ndarray
    origin_xyz: np.ndarray | None = None

    def __post_init__(self):
        vertices, rotation = np.asarray(self.vertices_xyz, float), np.asarray(self.rotation_matrix, float)
        if vertices.shape != (8, 3) or not np.isfinite(vertices).all():
            raise ValueError("body must contain all eight finite bounding-box vertices")
        if rotation.shape != (3, 3) or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5):
            raise ValueError("invalid body rotation matrix")
        object.__setattr__(self, "vertices_xyz", vertices.copy())
        object.__setattr__(self, "rotation_matrix", rotation.copy())
        origin = vertices.mean(axis=0) if self.origin_xyz is None else np.asarray(self.origin_xyz, float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("invalid body rotation origin")
        object.__setattr__(self, "origin_xyz", origin.copy())

    @property
    def polygon(self):
        return convex_hull(self.vertices_xyz[:, :2])

    @classmethod
    def from_actor(cls, actor, transform):
        vertices = actor.bounding_box.get_world_vertices(transform)
        return cls(int(actor.id), np.array([[v.x, v.y, v.z] for v in vertices]),
                   np.asarray(transform.get_matrix())[:3, :3],
                   np.array([transform.location.x, transform.location.y, transform.location.z]))


def swept_footprint(before, after):
    """Conservative envelope for linear translation and shortest 3D rotation.

    This is a 20 Hz interpolation assumption, not physics-substep telemetry.
    Radius*angle^2/8 bounds the rotational arc's distance from its chord.
    """
    if before.actor_id != after.actor_id:
        raise ValueError("cannot sweep different actors")
    angle = math.acos(float(np.clip((np.trace(before.rotation_matrix.T@after.rotation_matrix)-1)/2, -1, 1)))
    radius = max(np.linalg.norm(body.vertices_xyz-body.origin_xyz, axis=1).max()
                 for body in (before, after))
    polygon = convex_hull(np.vstack([before.vertices_xyz[:, :2], after.vertices_xyz[:, :2]]))
    return polygon, float(radius*angle*angle/8)


class LaneCorridor:
    def __init__(self, reference, ego_width_m, clearance_m=.15, step_m=.25):
        if not all(math.isfinite(v) and v > 0 for v in (ego_width_m, clearance_m, step_m)):
            raise ValueError("invalid lane corridor dimensions")
        # Include every spline knot. On each interval its second derivative is
        # linear, so the endpoint norm maximum gives a chord-error upper bound.
        arc = np.unique(np.r_[reference.s, np.arange(reference.s[0], reference.s[-1], step_m)])
        self.center_xy = reference.sample(arc)[0][:, :2]
        second = np.linalg.norm(reference._center(arc, 2)[:, :2], axis=1)
        self.chord_error_m = float(np.max(np.maximum(second[:-1], second[1:])*np.diff(arc)**2/8))
        self.half_width_m = ego_width_m/2+clearance_m

    def clearance(self, polygon, sweep_padding_m=0.):
        return polyline_polygon_distance(self.center_xy, polygon)-self.half_width_m-self.chord_error_m-sweep_padding_m


class TrafficValidator:
    def __init__(self, corridor, actor_ids, config=None):
        self.corridor, self.actor_ids = corridor, frozenset(actor_ids)
        self.config = config or TrafficConfig(count=len(self.actor_ids))
        if len(self.actor_ids) != self.config.count:
            raise ValueError("traffic actor identities do not match requested count")
        self.previous, self.last_frame, self.last_time = {}, None, None
        self.frames = 0
        self.reasons = set()
        self.minimum_corridor_clearance_m = math.inf
        self.minimum_body_clearance_m = math.inf

    def check(self, frame_id, timestamp_s, ego, traffic, collision_events=()):
        if type(frame_id) is not int or frame_id < 0 or not math.isfinite(timestamp_s):
            raise ValueError("traffic frame identity and time must be finite")
        if ego.actor_id in self.actor_ids:
            raise ValueError("ego cannot also be a surrounding traffic actor")
        reasons = []
        if self.last_frame is not None and (frame_id != self.last_frame+1 or
                not math.isclose(timestamp_s-self.last_time, .05, abs_tol=1e-5)):
            reasons.append("missing_duplicate_or_non20hz_traffic_frame")
        bodies = {body.actor_id: body for body in traffic}
        if set(bodies) != self.actor_ids or len(traffic) != len(self.actor_ids):
            reasons.append("missing_or_duplicate_traffic_actor")
        if collision_events:
            reasons.append("collision_event")
        all_bodies = {ego.actor_id: ego, **bodies}
        envelopes = {}
        for identity, body in all_bodies.items():
            envelopes[identity] = (body.polygon, 0.) if identity not in self.previous else swept_footprint(self.previous[identity], body)
        corridor_rows = []
        for identity in sorted(bodies):
            polygon, padding = envelopes[identity]
            clearance = self.corridor.clearance(polygon, padding)
            self.minimum_corridor_clearance_m = min(self.minimum_corridor_clearance_m, clearance)
            corridor_rows.append(dict(actor_id=identity, corridor_clearance_m=clearance, sweep_padding_m=padding))
            if clearance <= 0:
                reasons.append("traffic_body_intrudes_intended_corridor")
        identities = sorted(all_bodies)
        for i, first in enumerate(identities):
            for second in identities[i+1:]:
                a, ap = envelopes[first]; b, bp = envelopes[second]
                clearance = polygon_distance(a, b)-ap-bp
                self.minimum_body_clearance_m = min(self.minimum_body_clearance_m, clearance)
                if clearance < self.config.body_clearance_m:
                    reasons.append("traffic_body_contact_or_insufficient_clearance")
        self.previous, self.last_frame, self.last_time = all_bodies, int(frame_id), float(timestamp_s)
        self.frames += 1
        self.reasons.update(reasons)
        return dict(frame_id=int(frame_id), timestamp_s=float(timestamp_s), passed=not self.reasons,
                    reasons=sorted(self.reasons), actors=corridor_rows,
                    minimum_corridor_clearance_m=self.minimum_corridor_clearance_m if math.isfinite(self.minimum_corridor_clearance_m) else None,
                    minimum_body_clearance_m=self.minimum_body_clearance_m if math.isfinite(self.minimum_body_clearance_m) else None)


class SurroundingTraffic:
    """Call spawn/check/close inside the existing single world-owner context."""
    @classmethod
    def spawn(cls, client, world, ego, reference, *, worker, rpc_port, seed, config=None):
        import carla
        c = config or TrafficConfig()
        if worker not in (0, 1) or rpc_port != Config().workers[worker][1]:
            raise ValueError("traffic worker/RPC port mismatch")
        settings = world.get_settings()
        if not settings.synchronous_mode or not math.isclose(settings.fixed_delta_seconds or 0, .05, abs_tol=1e-8):
            raise ValueError("traffic requires the owner's synchronous 20 Hz world")
        self = cls()
        self.client, self.world, self.ego, self.config = client, world, ego, c
        self.worker, self.rpc_port, self.seed = worker, rpc_port, int(seed)
        self.tm_port = Config().workers[worker][2]
        self.vehicles, self.sensors, self.spawn_records, self.collisions = [], [], [], []
        self.events, self.tm, self.validator = queue.SimpleQueue(), None, None
        self.closed = False
        self.corridor = LaneCorridor(reference, 2*ego.bounding_box.extent.y, c.corridor_clearance_m, c.center_sample_step_m)
        try:
            self.tm = client.get_trafficmanager(self.tm_port)
            if self.tm.get_port() != self.tm_port:
                raise RuntimeError("traffic manager port mismatch")
            self.tm.set_synchronous_mode(True)
            self.tm.set_random_device_seed(int(seed))
            self.tm.set_hybrid_physics_mode(False)
            self.tm.set_respawn_dormant_vehicles(False)
            road_map = world.get_map()
            candidates, seen = [], set()
            for arc in np.arange(reference.s[0], reference.s[-1], c.candidate_step_m):
                point, yaw, _, _ = reference.sample(arc)
                base = road_map.get_waypoint(carla.Location(*map(float, point)), project_to_road=True, lane_type=carla.LaneType.Driving)
                if base is None or base.is_junction or base.transform.location.distance(carla.Location(*map(float, point))) > 1.:
                    continue
                if math.cos(math.radians(base.transform.rotation.yaw)-float(yaw)) < .95:
                    continue
                for adjacent in (base.get_left_lane(), base.get_right_lane()):
                    if adjacent is None or adjacent.lane_type != carla.LaneType.Driving or adjacent.is_junction:
                        continue
                    if adjacent.road_id != base.road_id or adjacent.lane_id*base.lane_id <= 0:
                        continue
                    if math.cos(math.radians(adjacent.transform.rotation.yaw-base.transform.rotation.yaw)) < .95:
                        continue
                    key = (adjacent.road_id, adjacent.section_id, adjacent.lane_id, round(adjacent.s, 2))
                    if key not in seen:
                        seen.add(key); candidates.append(adjacent)
            random.Random(seed).shuffle(candidates)
            ego_body = BodyFootprint.from_actor(ego, ego.get_transform())
            initial_bodies = []
            for waypoint in candidates[:c.maximum_candidates]:
                transform = waypoint.transform
                transform.location.z += .25
                blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
                blueprint.set_attribute("role_name", f"lane_dataset_traffic_w{worker}")
                vehicle = world.try_spawn_actor(blueprint, transform)
                if vehicle is None:
                    continue
                self.vehicles.append(vehicle)
                body = BodyFootprint.from_actor(vehicle, transform)
                if (self.corridor.clearance(body.polygon) <= 0 or
                    polygon_distance(body.polygon, ego_body.polygon) < c.spawn_ego_clearance_m or
                    any(polygon_distance(body.polygon, other.polygon) < c.spawn_traffic_clearance_m for other in initial_bodies)):
                    vehicle.destroy(); self.vehicles.pop(); continue
                vehicle.set_simulate_physics(True)
                vehicle.set_autopilot(True, self.tm_port)
                self.tm.auto_lane_change(vehicle, False)
                self.tm.random_left_lanechange_percentage(vehicle, 0.)
                self.tm.random_right_lanechange_percentage(vehicle, 0.)
                self.tm.distance_to_leading_vehicle(vehicle, 8.)
                self.tm.set_desired_speed(vehicle, c.speed_mps*3.6)
                self.tm.ignore_lights_percentage(vehicle, 0.)
                self.tm.ignore_signs_percentage(vehicle, 0.)
                self.tm.ignore_vehicles_percentage(vehicle, 0.)
                self.tm.ignore_walkers_percentage(vehicle, 0.)
                self.tm.collision_detection(vehicle, ego, True)
                self.tm.update_vehicle_lights(vehicle, True)
                initial_bodies.append(body)
                self.spawn_records.append(dict(actor_id=vehicle.id, blueprint=blueprint.id, road_id=waypoint.road_id,
                    lane_id=waypoint.lane_id, road_s=waypoint.s, position=[transform.location.x, transform.location.y, transform.location.z],
                    initial_corridor_clearance_m=self.corridor.clearance(body.polygon)))
                if len(self.vehicles) == c.count:
                    break
            if len(self.vehicles) != c.count:
                raise TrafficRejected(dict(reasons=[f"insufficient_safe_adjacent_traffic_spawns:{len(self.vehicles)}/{c.count}"]))
            for vehicle in [ego, *self.vehicles]:
                sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=vehicle)
                self.sensors.append(sensor)
                sensor.listen(lambda event, identity=vehicle.id: self.events.put(dict(actor_id=identity,
                    other_actor_id=event.other_actor.id, frame_id=int(event.frame), timestamp_s=float(event.timestamp),
                    impulse=[event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z])))
            self.validator = TrafficValidator(self.corridor, [v.id for v in self.vehicles], c)
            return self
        except BaseException:
            self.close()
            raise

    def _drain_collisions(self):
        while True:
            try:
                self.collisions.append(self.events.get_nowait())
            except queue.Empty:
                return

    def check(self, snapshot, *, raise_on_failure=True):
        if self.closed:
            raise RuntimeError("traffic session already closed")
        self._drain_collisions()
        bodies = []
        for vehicle in [self.ego, *self.vehicles]:
            state = snapshot.find(vehicle.id)
            if state is None:
                report = dict(frame_id=int(snapshot.frame), passed=False, reasons=["traffic_actor_missing_from_snapshot"])
                self.validator.reasons.update(report["reasons"])
                if raise_on_failure:
                    raise TrafficRejected(report)
                return report
            bodies.append(BodyFootprint.from_actor(vehicle, state.get_transform()))
        report = self.validator.check(snapshot.frame, snapshot.timestamp.elapsed_seconds, bodies[0], bodies[1:], self.collisions)
        if not report["passed"] and raise_on_failure:
            raise TrafficRejected(report)
        return report

    def summary(self):
        self._drain_collisions()
        reasons = set() if self.validator is None else set(self.validator.reasons)
        if self.collisions:
            reasons.add("collision_event")
        frames = 0 if self.validator is None else self.validator.frames
        if frames == 0:
            reasons.add("no_validated_traffic_frames")
        if len(self.spawn_records) != self.config.count:
            reasons.add("traffic_spawn_count_mismatch")
        return dict(requested_count=self.config.count, spawned_count=len(self.spawn_records), worker=self.worker,
            rpc_port=self.rpc_port, traffic_manager_port=self.tm_port, seed=self.seed,
            physics=True, hybrid_physics=False, synchronous=True, automatic_lane_change=False,
            desired_speed_mps=self.config.speed_mps, spawn_records=self.spawn_records,
            validated_frames=frames, collisions=list(self.collisions), passed=not reasons, reasons=sorted(reasons),
            corridor_half_width_m=self.corridor.half_width_m, chord_error_bound_m=self.corridor.chord_error_m,
            minimum_corridor_clearance_m=None if not frames or not math.isfinite(self.validator.minimum_corridor_clearance_m) else self.validator.minimum_corridor_clearance_m,
            minimum_body_clearance_m=None if not frames or not math.isfinite(self.validator.minimum_body_clearance_m) else self.validator.minimum_body_clearance_m,
            sweep_assumption="linear_translation_and_shortest_rotation_between_20hz_snapshots",
            scope="surrounding_traffic_obstruction_QA_not_expert_or_corpus_qualification")

    def close(self):
        if self.closed:
            return []
        errors = []
        for sensor in self.sensors:
            try:
                sensor.stop(); sensor.destroy()
            except RuntimeError as error:
                errors.append(str(error))
        for vehicle in self.vehicles:
            try:
                vehicle.set_autopilot(False, self.tm_port); vehicle.destroy()
            except RuntimeError as error:
                errors.append(str(error))
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except RuntimeError as error:
                errors.append(str(error))
        self._drain_collisions()
        self.closed = True
        return errors
