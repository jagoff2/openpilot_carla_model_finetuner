"""Deterministic geometry grouping and fail-closed corpus release accounting."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence


FAMILIES = ("straight_recovery", "constant_radius", "tightening_s_bend",
            "severe_hairpin", "intersection_turn")
SPLITS = ("train", "validation", "test")
LABEL_QUOTAS = {"train": 4000, "validation": 500, "test": 500}
GROUP_QUOTAS = {"train": 100, "validation": 25, "test": 25}
EPISODES_PER_GROUP = {"train": 2, "validation": 1, "test": 1}
LABELS_PER_EPISODE = 20
APPROVED_TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RoadInterval:
    road_id: str
    start_m: float
    end_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_m) or not math.isfinite(self.end_m):
            raise ValueError("road interval must be finite")
        # A reversed traversal occupies the same physical road interval.
        start, end = sorted((float(self.start_m), float(self.end_m)))
        object.__setattr__(self, "road_id", str(self.road_id))
        object.__setattr__(self, "start_m", start)
        object.__setattr__(self, "end_m", end)


@dataclass(frozen=True)
class GeometrySection:
    section_id: str
    town: str
    family: str
    road_intervals: tuple[RoadInterval, ...] = ()
    junction_ids: tuple[str, ...] = ()
    # Links must come from the road inventory, including junctions absent from
    # this particular traversal. Otherwise split leakage can escape detection.
    junction_links: tuple[tuple[str, str], ...] = ()
    procedural: bool = False
    procedural_parameters: Mapping[str, float | str] = field(default_factory=dict)
    # Same geometry may have a different road ID after regeneration. This key
    # must be computed before appearance, traversal direction, seed, or split.
    geometry_key: str | None = None
    direction: str = "straight"

    def __post_init__(self) -> None:
        if not self.section_id or self.family not in FAMILIES:
            raise ValueError("section_id and a recognized scenario family are required")
        if not self.procedural and self.town not in APPROVED_TOWNS:
            raise ValueError("native geometry must use an approved map")
        if self.procedural and not self.procedural_parameters:
            raise ValueError("procedural geometry requires canonical parameters")
        if self.direction not in ("left", "right", "straight"):
            raise ValueError("direction must be left, right, or straight")
        # CARLA IDs arrive as integers through the API and strings from JSON.
        # Normalize once so candidate graph and final regrouping agree.
        object.__setattr__(self, "junction_ids", tuple(str(value) for value in self.junction_ids))
        object.__setattr__(self, "junction_links", tuple((str(left),str(right)) for left,right in self.junction_links))


@dataclass(frozen=True)
class GeometryGroup:
    group_id: str
    sections: tuple[GeometrySection, ...]

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({section.family for section in self.sections}))

    @property
    def towns(self) -> tuple[str, ...]:
        return tuple(sorted({section.town for section in self.sections if not section.procedural}))

    @property
    def procedural(self) -> bool:
        return any(section.procedural for section in self.sections)


@dataclass(frozen=True)
class SplitAssignment:
    group_id: str
    split: str
    family: str
    appearances: int
    seed: int


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, one: str, two: str) -> None:
        left, right = self.find(one), self.find(two)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def procedural_signature(section: GeometrySection) -> str:
    # Exact parameters are recorded separately. The same procedural geometry
    # has one key even if named differently or rendered with another appearance.
    parameters=dict(section.procedural_parameters)
    # Mirrors and split names never create an independent physical group.
    # Scenario-family annotations likewise cannot distinguish identical shape
    # parameters. The canonical parameters must contain the generating shape.
    parameters.pop("turn",None)
    parameters.pop("split",None)
    return _stable_hash(parameters)


def group_geometry_sections(sections: Sequence[GeometrySection], *,
                            overlap_margin_m: float = 5.0) -> tuple[GeometryGroup, ...]:
    """Union overlap, reverse traversals, connected junctions and appearances.

    Road IDs are namespaced by town; lane direction is deliberately excluded.
    The overlap margin protects adjacent intervals from sharing preview pixels.
    Supply the entire observed route envelope, including pre/post-roll, rather
    than only the 20 labeled positions.
    """
    if overlap_margin_m < 0 or not math.isfinite(overlap_margin_m):
        raise ValueError("overlap margin must be finite and nonnegative")
    ids = [section.section_id for section in sections]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate section ids")
    geometry = _UnionFind()
    junctions = _UnionFind()
    for section in sections:
        geometry.find(section.section_id)
        for left, right in section.junction_links:
            junctions.union(f"{section.town}:{left}", f"{section.town}:{right}")
        for junction in section.junction_ids:
            junctions.find(f"{section.town}:{junction}")
    roads: dict[tuple[str, str], list[tuple[RoadInterval, str]]] = defaultdict(list)
    shared_keys: dict[str, str] = {}
    for section in sections:
        keys = [f"junction:{junctions.find(f'{section.town}:{junction}')}"
                for junction in section.junction_ids]
        if section.geometry_key:
            keys.append(f"geometry:{section.geometry_key}")
        if section.procedural:
            keys.append(f"procedural:{procedural_signature(section)}")
        for key in keys:
            if key in shared_keys:
                geometry.union(section.section_id, shared_keys[key])
            else:
                shared_keys[key] = section.section_id
        for interval in section.road_intervals:
            roads[(section.town, interval.road_id)].append((interval, section.section_id))
    for intervals in roads.values():
        intervals.sort(key=lambda value: (value[0].start_m, value[0].end_m, value[1]))
        # Sweep maintaining the furthest end of the current connected interval.
        component_id = None
        component_end = -math.inf
        for interval, section_id in intervals:
            if interval.start_m <= component_end + overlap_margin_m and component_id is not None:
                geometry.union(component_id, section_id)
                component_end = max(component_end, interval.end_m)
            else:
                component_id, component_end = section_id, interval.end_m
    members: dict[str, list[GeometrySection]] = defaultdict(list)
    for section in sections:
        members[geometry.find(section.section_id)].append(section)
    groups = []
    for values in members.values():
        values.sort(key=lambda section: section.section_id)
        content = [asdict(section) for section in values]
        groups.append(GeometryGroup("geom-" + _stable_hash(content)[:20], tuple(values)))
    return tuple(sorted(groups, key=lambda group: group.group_id))


def assign_splits(groups: Sequence[GeometryGroup], *, seed: int = 20261013,
                  group_quotas: Mapping[str, int] = GROUP_QUOTAS) -> tuple[SplitAssignment, ...]:
    """Select fixed balanced groups without ever splitting a geometry group.

    Insufficient native coverage is a preparation failure. It is not repaired
    by changing a family, moving Town05 into training, or duplicating groups.
    Procedural selection is capped at 30% of groups in each split.
    """
    if set(group_quotas) != set(SPLITS) or any(value <= 0 or value % len(FAMILIES) for value in group_quotas.values()):
        raise ValueError("group quotas must be positive multiples of the five families")
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("duplicate geometry group ids")
    by_id = {group.group_id: group for group in groups}
    used: set[str] = set()
    assignments: list[SplitAssignment] = []
    procedural_counts = Counter()
    # Reserve native Town05 first. Then reserve validation before training to
    # avoid consuming scarce holdout geometry with two appearance variants.
    for split in ("test", "validation", "train"):
        per_family = group_quotas[split] // len(FAMILIES)
        # Most constrained family first; deterministic seeded ordering within.
        candidate_map = {}
        for family in FAMILIES:
            eligible = [group for group in groups if family in group.families and group.group_id not in used
                        and ((not group.towns or group.towns == ("Town05",)) if split == "test"
                             else "Town05" not in group.towns)]
            candidate_map[family] = sorted(eligible, key=lambda group: (
                group.procedural, _stable_hash([seed, split, family, group.group_id])))
        for family in sorted(FAMILIES, key=lambda value: (len(candidate_map[value]), value)):
            selected = 0
            for group in candidate_map[family]:
                if group.group_id in used:
                    continue
                if group.procedural and procedural_counts[split] + 1 > math.floor(group_quotas[split] * .30):
                    continue
                used.add(group.group_id)
                procedural_counts[split] += int(group.procedural)
                episode_seed = int(_stable_hash([seed, group.group_id, split])[:8], 16)
                assignments.append(SplitAssignment(group.group_id, split, family,
                                                   EPISODES_PER_GROUP[split], episode_seed))
                selected += 1
                if selected == per_family:
                    break
            if selected != per_family:
                raise ValueError(f"insufficient independent {split}/{family} geometry groups: "
                                 f"{selected}/{per_family}; quotas remain unchanged")
    assert all(assignment.group_id in by_id for assignment in assignments)
    return tuple(sorted(assignments, key=lambda item: (SPLITS.index(item.split), item.family, item.group_id)))


def audit_split_overlap(groups: Sequence[GeometryGroup], assignments: Sequence[SplitAssignment], *,
                         overlap_margin_m: float = 5.0) -> list[str]:
    """Recompute grouping from raw geometry to detect manipulated group ids."""
    errors: list[str] = []
    assignment_by_group: dict[str, str] = {}
    for assignment in assignments:
        if assignment.group_id in assignment_by_group:
            errors.append(f"duplicate_assignment:{assignment.group_id}")
        assignment_by_group[assignment.group_id] = assignment.split
    known = {group.group_id for group in groups}
    for missing in set(assignment_by_group) - known:
        errors.append(f"assignment_unknown_group:{missing}")
    section_split: dict[str, str] = {}
    sections: list[GeometrySection] = []
    for group in groups:
        split = assignment_by_group.get(group.group_id)
        if split is None:
            continue
        for section in group.sections:
            if section.section_id in section_split:
                errors.append(f"duplicate_section:{section.section_id}")
                continue
            sections.append(section)
            section_split[section.section_id] = split
            if section.town == "Town05" and split != "test" and not section.procedural:
                errors.append(f"town05_outside_test:{section.section_id}")
            if split == "test" and section.town != "Town05" and not section.procedural:
                errors.append(f"native_test_outside_town05:{section.section_id}")
    for group in group_geometry_sections(sections, overlap_margin_m=overlap_margin_m):
        split_set = {section_split[section.section_id] for section in group.sections}
        if len(split_set) > 1:
            errors.append(f"geometry_leakage:{group.group_id}:{','.join(sorted(split_set))}")
    return sorted(set(errors))


def audit_crop_loss_evidence(record: Mapping) -> list[str]:
    """Independently check the compact pixel ledger before counting crop loss.

    Archive validation must separately resolve references to authenticated image
    frames. A category string or nominal recurrent-buffer length is insufficient.
    """
    value = record.get("visibility")
    if not isinstance(value, Mapping):
        return ["missing_visibility_ledger"]
    if value.get("category") != "history_crop_loss":
        return ["visibility_category_mismatch"]
    required = ("point_ids", "distance_m", "required_mask", "supported", "current_visible",
                "current_source_visible", "evidence_frames", "visibility_duration_s")
    if any(key not in value for key in required):
        return ["incomplete_visibility_ledger"]
    try:
        size = len(value["point_ids"])
        if size == 0 or any(len(value[key]) != size for key in required):
            return ["visibility_ledger_shape_mismatch"]
        timestamp = float(record["timestamp_s"])
        current_preview = float(value["preview_distance_m"])
        preview_loss = float(value["crop_preview_loss_m"])
        if not all(math.isfinite(number) for number in (timestamp,current_preview,preview_loss)):
            return ["nonfinite_visibility_metrics"]
        if preview_loss < 2.0 - 1e-8:
            return ["insufficient_measured_crop_preview_loss"]
        qualified_points = []
        for index in range(size):
            refs = value["evidence_frames"][index]
            for ref in refs:
                if (ref["camera"] not in ("narrow","wide") or not math.isfinite(float(ref["timestamp_s"]))
                        or float(ref["timestamp_s"]) > timestamp + 1e-8
                        or float(ref["timestamp_s"]) < timestamp - 4.8 - 1e-8):
                    return ["noncausal_or_expired_visibility_reference"]
            if not (value["required_mask"][index] and value["supported"][index]
                    and not value["current_visible"][index] and value["current_source_visible"][index]):
                continue
            if not refs or any(float(ref["timestamp_s"]) >= timestamp - 1e-8 for ref in refs):
                return ["crop_missing_point_without_exclusively_historical_evidence"]
            times = sorted({float(ref["timestamp_s"]) for ref in refs})
            duration = sum(b-a for a,b in zip(times,times[1:]) if 0 < b-a <= .050001)
            recorded_duration = float(value["visibility_duration_s"][index])
            if not math.isfinite(recorded_duration) or abs(duration-recorded_duration) > 1e-6:
                return ["visibility_duration_mismatch"]
            distance = float(value["distance_m"][index])
            if not math.isfinite(distance):
                return ["nonfinite_visibility_distance"]
            if duration >= .1 - 1e-8 and distance > 0:
                qualified_points.append(distance)
        if not qualified_points or max(qualified_points)-current_preview < 2.0 - 1e-8:
            return ["no_qualified_historical_curve_preview"]
        if abs(preview_loss - (max(qualified_points)-current_preview)) > 1e-6:
            return ["crop_preview_loss_mismatch"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return ["invalid_visibility_ledger"]
    return []


def audit_release(labels: Iterable[Mapping], groups: Sequence[GeometryGroup],
                  assignments: Sequence[SplitAssignment], *,
                  require_reviews: bool = True) -> dict:
    """Audit one manifest record per accepted or diagnostic decision timestamp.

    Accepted records require split, family, group_id, episode_id, label_index,
    timestamp_s, appearance, procedural, direction, visibility_category,
    action_valid, qa_passed, and reviewed. Evidence is additionally checked by
    the visibility module and shard validator. Diagnostics never count toward
    any accepted quota, even if they carry an ``accepted`` flag accidentally.
    """
    errors = audit_split_overlap(groups, assignments)
    group_lookup = {group.group_id: group for group in groups}
    assignment_lookup = {item.group_id: item for item in assignments}
    counts = Counter()
    family_counts = Counter()
    crop_counts = Counter()
    procedural_counts = Counter()
    directions = Counter()
    diagnostic_count = 0
    accepted: list[Mapping] = []
    keys: set[tuple[str, int]] = set()
    episode_labels: dict[str, list[Mapping]] = defaultdict(list)
    group_episodes: dict[str, set[str]] = defaultdict(set)
    required_fields = {"split", "family", "group_id", "episode_id", "label_index", "timestamp_s",
                       "appearance", "procedural", "direction", "visibility_category", "action_valid", "qa_passed"}
    for record in labels:
        if not record.get("accepted", False):
            diagnostic_count += int(record.get("visibility_category") == "diagnostic_ambiguous")
            continue
        missing = required_fields - record.keys()
        if missing:
            errors.append(f"accepted_record_missing_fields:{','.join(sorted(missing))}")
            continue
        if record["visibility_category"] == "diagnostic_ambiguous":
            errors.append(f"ambiguous_record_marked_accepted:{record['episode_id']}:{record['label_index']}")
            diagnostic_count += 1
            continue
        split, family, group_id = record["split"], record["family"], record["group_id"]
        episode_id = str(record["episode_id"])
        if split not in SPLITS or family not in FAMILIES:
            errors.append(f"invalid_split_or_family:{episode_id}")
            continue
        if group_id not in group_lookup or group_id not in assignment_lookup:
            errors.append(f"accepted_unknown_group:{group_id}")
            continue
        assignment = assignment_lookup[group_id]
        if assignment.split != split or assignment.family != family:
            errors.append(f"accepted_assignment_mismatch:{episode_id}")
        if bool(record["procedural"]) != group_lookup[group_id].procedural:
            errors.append(f"procedural_flag_mismatch:{episode_id}")
        if not record["action_valid"] or not record["qa_passed"]:
            errors.append(f"accepted_failed_gate:{episode_id}:{record['label_index']}")
        if require_reviews and not record.get("reviewed", False):
            errors.append(f"unreviewed_geometry:{group_id}")
        key = (episode_id, int(record["label_index"]))
        if key in keys:
            errors.append(f"duplicate_label:{episode_id}:{record['label_index']}")
            continue
        keys.add(key)
        counts[split] += 1
        family_counts[(split, family)] += 1
        if record["visibility_category"] == "history_crop_loss":
            evidence_errors = audit_crop_loss_evidence(record)
            if evidence_errors:
                errors.extend(f"crop_evidence:{episode_id}:{record['label_index']}:{reason}"
                              for reason in evidence_errors)
            else:
                crop_counts[split] += 1
        procedural_counts[split] += int(record["procedural"])
        directions[(split, family, record["direction"])] += 1
        episode_labels[episode_id].append(record)
        group_episodes[group_id].add(episode_id)
        accepted.append(record)
    for split in SPLITS:
        if counts[split] != LABEL_QUOTAS[split]:
            errors.append(f"label_quota:{split}:{counts[split]}/{LABEL_QUOTAS[split]}")
        for family in FAMILIES:
            target = LABEL_QUOTAS[split] // len(FAMILIES)
            if family_counts[(split, family)] != target:
                errors.append(f"family_quota:{split}/{family}:{family_counts[(split, family)]}/{target}")
            if family != "straight_recovery":
                left = directions[(split, family, "left")]
                right = directions[(split, family, "right")]
                # Five 20-label holdout episodes are indivisible for a constant
                # left/right bend: 40/60 is the closest possible balance. Keep
                # the discrepancy at most one episode rather than inventing a
                # sixth geometry group or changing the 20-label episode rule.
                tolerance = 0 if split == "train" else LABELS_PER_EPISODE
                if abs(left - right) > tolerance or left + right != family_counts[(split, family)]:
                    errors.append(f"left_right_balance:{split}/{family}:{left}/{right}")
        if crop_counts[split] < math.ceil(LABEL_QUOTAS[split] * .25):
            errors.append(f"crop_history_quota:{split}:{crop_counts[split]}/{math.ceil(LABEL_QUOTAS[split] * .25)}")
        if procedural_counts[split] > math.floor(LABEL_QUOTAS[split] * .30):
            errors.append(f"procedural_cap:{split}:{procedural_counts[split]}/{math.floor(LABEL_QUOTAS[split] * .30)}")
        actual_groups = sum(1 for group_id in group_episodes if assignment_lookup[group_id].split == split)
        if actual_groups != GROUP_QUOTAS[split]:
            errors.append(f"geometry_group_quota:{split}:{actual_groups}/{GROUP_QUOTAS[split]}")
    for episode_id, records in episode_labels.items():
        if len(records) != LABELS_PER_EPISODE:
            errors.append(f"episode_label_quota:{episode_id}:{len(records)}/{LABELS_PER_EPISODE}")
        for name in ("group_id", "split", "family", "appearance"):
            if len({str(record[name]) for record in records}) != 1:
                errors.append(f"episode_inconsistent_{name}:{episode_id}")
        timestamps = [float(record["timestamp_s"]) for record in records]
        origin = float(records[0].get("candidate_grid_origin_s", min(timestamps)))
        if not all(math.isfinite(value) and abs((value-origin)*2-round((value-origin)*2)) < 1e-6 for value in timestamps):
            errors.append(f"off_candidate_grid:{episode_id}")
        if len(set(timestamps)) != len(timestamps):
            errors.append(f"duplicate_decision_timestamp:{episode_id}")
    for group_id, episode_ids in group_episodes.items():
        assignment = assignment_lookup[group_id]
        expected = EPISODES_PER_GROUP[assignment.split]
        if len(episode_ids) != expected:
            errors.append(f"group_appearance_quota:{group_id}:{len(episode_ids)}/{expected}")
        appearances = {str(episode_labels[episode_id][0]["appearance"]) for episode_id in episode_ids}
        if len(appearances) != expected:
            errors.append(f"group_appearance_not_distinct:{group_id}")
    return {"qualified": not errors, "errors": sorted(set(errors)),
            "accepted_counts": {split: counts[split] for split in SPLITS},
            "family_counts": {split: {family: family_counts[(split, family)] for family in FAMILIES}
                              for split in SPLITS},
            "crop_history_counts": {split: crop_counts[split] for split in SPLITS},
            "procedural_counts": {split: procedural_counts[split] for split in SPLITS},
            "diagnostic_ambiguous_count": diagnostic_count,
            "direction_balance_granularity_labels": {"train": 0, "validation": 20, "test": 20},
            "accepted_total": len(accepted)}
