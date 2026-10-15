"""Explicit, source-bound permission for a controlled static parked-car pass.

Permission is neither a declaration that hidden traffic is absent nor a new
driving center. Observed permission requires independently replayed visible
road surface AND a separate controlled-scene inventory artifact. Consumers
must independently admit those artifacts before publishing training records.
"""
from dataclasses import asdict,dataclass,replace
import hashlib
import json

import numpy as np

from lane_dataset.geometry import EvidenceFrame


OBSERVED_VERSION='observed_controlled_static_parked_pass_v1'
PLANNED_VERSION='privileged_controlled_static_parked_pass_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def reference_fingerprint(reference):
    return _digest(dict(left_xyz=np.asarray(reference.left_xyz).tolist(),
        right_xyz=np.asarray(reference.right_xyz).tolist(),s=np.asarray(reference.s).tolist()))


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _validate_fields(value):
    if not _sha(value.reference_sha256) or not _sha(value.controlled_scene_sha256):
        raise ValueError('passing permission requires immutable reference and controlled-scene SHA256 bindings')
    intervals=np.asarray(value.allowed_intervals_m,float)
    if (intervals.ndim!=2 or intervals.shape[1]!=2 or not len(intervals)
            or not np.isfinite(intervals).all() or np.any(intervals[:,1]<=intervals[:,0])
            or np.any(intervals[1:,0]<intervals[:-1,1])):
        raise ValueError('passing permission requires sorted disjoint finite observed station intervals')
    if (isinstance(value.opposing_width_m,bool) or not np.isfinite(value.opposing_width_m)
            or value.opposing_width_m<=0):
        raise ValueError('passing width must be explicit positive observed space')
    if (not value.obstacle_ids or len(set(value.obstacle_ids))!=len(value.obstacle_ids)
            or any(not isinstance(k,str) or not k for k in value.obstacle_ids)):
        raise ValueError('passing permission requires distinct designated parked-car identities')
    object.__setattr__(value,'allowed_intervals_m',tuple(tuple(float(q) for q in row) for row in intervals))
    object.__setattr__(value,'obstacle_ids',tuple(value.obstacle_ids))
    object.__setattr__(value,'opposing_width_m',float(value.opposing_width_m))


@dataclass(frozen=True)
class PlannedPassingSpace:
    reference_sha256:str
    allowed_intervals_m:tuple
    opposing_width_m:float
    obstacle_ids:tuple[str,...]
    controlled_scene_sha256:str
    version:str=PLANNED_VERSION

    def __post_init__(self):
        _validate_fields(self)
        if self.version!=PLANNED_VERSION:raise ValueError('invalid privileged passing-space version')

    def document(self):return asdict(self)

    @classmethod
    def from_document(cls,value):return cls(**value)


@dataclass(frozen=True)
class ObservedPassingSpace:
    reference_sha256:str
    allowed_intervals_m:tuple
    opposing_width_m:float
    obstacle_ids:tuple[str,...]
    controlled_scene_sha256:str
    rendered_free_space_sha256:str
    evidence_frames:tuple[EvidenceFrame,...]
    version:str=OBSERVED_VERSION

    def __post_init__(self):
        _validate_fields(self)
        if (self.version!=OBSERVED_VERSION or not _sha(self.rendered_free_space_sha256)
                or not self.evidence_frames or any(not isinstance(e,EvidenceFrame) for e in self.evidence_frames)):
            raise ValueError('observed passing space requires explicit causal rendered-surface proof')
        object.__setattr__(self,'evidence_frames',tuple(self.evidence_frames))

    def document(self):return asdict(self)

    @classmethod
    def from_document(cls,value):
        doc=dict(value);doc['evidence_frames']=tuple(EvidenceFrame(**e) for e in doc['evidence_frames'])
        return cls(**doc)


def permission_issue(space,reference,obstacles,*,observed,maximum_age_s=5.):
    if space is None:return None
    expected=ObservedPassingSpace if observed else PlannedPassingSpace
    if not isinstance(space,expected):return 'passing_space_provenance_mismatch'
    if space.reference_sha256!=reference_fingerprint(reference):return 'passing_space_reference_mismatch'
    if any(lo<reference.s[0]-1e-8 or hi>reference.s[-1]+1e-8 for lo,hi in space.allowed_intervals_m):
        return 'passing_space_outside_observed_reference'
    by_id={o.identity:o for o in obstacles}
    if any(name not in by_id for name in space.obstacle_ids):return 'passing_space_designated_obstacle_missing'
    if any(by_id[name].kind!='parked_vehicle' or by_id[name].motion!='static' for name in space.obstacle_ids):
        return 'passing_space_requires_static_parked_vehicle'
    if observed:
        now=reference.decision_time_s
        for e in space.evidence_frames:
            if (not isinstance(e.frame_id,int) or isinstance(e.frame_id,bool) or e.frame_id<0
                    or not np.isfinite(e.timestamp_s) or e.timestamp_s>now+1e-8
                    or e.camera not in ('narrow','wide','img','big_img')):
                return 'passing_space_missing_or_future_evidence'
            if now-e.timestamp_s>maximum_age_s+1e-8:return 'passing_space_stale_evidence'
    return None


def transformed_observed_space(space,source_reference,local_reference):
    """After replaying a world record, retain proofs through its rigid transform."""
    if space is None:return None
    if not isinstance(space,ObservedPassingSpace) or space.reference_sha256!=reference_fingerprint(source_reference):
        raise ValueError('cannot transform unbound observed passing space')
    if not np.array_equal(source_reference.s,local_reference.s):
        raise ValueError('rigid passing-space transform cannot remap station intervals')
    return replace(space,reference_sha256=reference_fingerprint(local_reference))


def permitted_edges(reference,query,excursions=(),passing_space=None):
    q=np.asarray(query,float)
    left=reference._left(q)[...,:2];right=reference._right(q)[...,:2]
    divider=(left+right)*.5
    if passing_space is None:return divider,right
    active=np.zeros(q.shape,bool)
    for event in excursions:
        if set(event.obstacle_ids).issubset(passing_space.obstacle_ids):
            active|=(q>=event.start_s)&(q<=event.end_s)
    observed=np.zeros(q.shape,bool)
    for lo,hi in passing_space.allowed_intervals_m:observed|=(q>=lo)&(q<=hi)
    across=right-left;half_width=np.linalg.norm(across,axis=-1)*.5
    width=np.minimum(passing_space.opposing_width_m,half_width)
    permitted=divider-across/np.maximum(2.*half_width[...,None],1e-12)*width[...,None]
    return np.where((active&observed)[...,None],permitted,divider),right


def permitted_ribbon(reference,spacing,excursions=(),passing_space=None):
    # Include both sides of each permission discontinuity explicitly. The
    # resulting polygon is the declared permitted corridor, not an expanded
    # spline interpolating through a sudden full-width lane-change boundary.
    base=np.unique(np.r_[reference.s,np.arange(reference.s[0],reference.s[-1],spacing)])
    if passing_space is not None:
        joins=[q for e in excursions for q in (e.start_s,e.end_s)]
        joins += [q for pair in passing_space.allowed_intervals_m for q in pair]
        joins=np.asarray(joins,float)
        joins=joins[(joins>reference.s[0])&(joins<reference.s[-1])]
        base=np.unique(np.r_[base,joins,joins-1e-7,joins+1e-7])
        base=base[(base>=reference.s[0])&(base<=reference.s[-1])]
    left,right=permitted_edges(reference,base,excursions,passing_space)
    return np.concatenate((left,right[::-1]))
