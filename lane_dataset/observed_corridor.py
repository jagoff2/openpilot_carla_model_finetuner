"""Build both sides from their own visible knots, without hidden map geometry."""
import numpy as np
from scipy.interpolate import CubicSpline
from .geometry import ObservableReference, MAX_OBSERVED_KNOT_GAP_M

METHOD='independent_observed_boundary_splines_on_sample_identity_no_hidden_map_derivatives'


def reconstruct(reference, support, position, timestamp, maximum_gap_m=MAX_OBSERVED_KNOT_GAP_M):
    n=len(reference.s)
    if len(support.supported)!=2*n or not np.isfinite(maximum_gap_m) or maximum_gap_m<=0:
        raise ValueError('invalid independent boundary support')
    runs=[]
    for coordinates,mask in ((reference.left_xyz,support.supported[:n]),(reference.right_xyz,support.supported[n:])):
        observed=np.flatnonzero(mask)
        gaps=np.linalg.norm(np.diff(coordinates[observed],axis=0),axis=1)
        runs.append([run for run in np.split(observed,np.flatnonzero(gaps>maximum_gap_m)+1) if len(run)>=4])
    if any(not side for side in runs):raise ValueError('insufficient_independent_boundary_observations')
    candidates=[]
    for left in runs[0]:
        for right in runs[1]:
            lo=max(left[0],right[0]);hi=min(left[-1],right[-1])
            grid=np.union1d(left,right);grid=grid[(grid>=lo)&(grid<=hi)]
            if len(grid)<4:continue
            # Sample identities supply correspondence, not distance or hidden
            # coordinates. Only visible XYZ values enter either boundary fit.
            # There is no extrapolation beyond either side's observed extent.
            l=CubicSpline(left,reference.left_xyz[left],axis=0)(grid)
            r=CubicSpline(right,reference.right_xyz[right],axis=0)(grid)
            centers=(l+r)*.5
            score=float(np.min(np.linalg.norm(centers[:,:2]-np.asarray(position)[:2],axis=1)))
            candidates.append((score,left,right,grid,l,r))
    if not candidates:raise ValueError('observed_boundaries_have_no_common_corridor')
    _,left,right,grid,l,r=min(candidates,key=lambda value:value[0])
    source_points=np.r_[left,right+n]
    refs=tuple(dict.fromkeys(ref for i in source_points for ref in support.evidence_frames[i]))
    if any(not support.evidence_frames[i] for i in source_points):raise ValueError('observed boundary knot lacks evidence')
    centers=(l+r)*.5;chord=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))]
    result=ObservableReference(l,r,chord,[refs]*len(grid),timestamp)
    result.source_indices=grid
    result.source_point_indices=source_points
    result.source_left_indices=left
    result.source_right_indices=right
    result.maximum_marking_gap_m=maximum_gap_m
    result.support_method=METHOD
    return result


def provenance(reference):
    return dict(source_indices=reference.source_indices.tolist(),
        source_point_indices=reference.source_point_indices.tolist(),
        source_left_indices=reference.source_left_indices.tolist(),
        source_right_indices=reference.source_right_indices.tolist(),
        chord_arc_m=reference.s.tolist(),maximum_marking_gap_m=reference.maximum_marking_gap_m,
        support_method=METHOD)
