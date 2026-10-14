from types import SimpleNamespace
import numpy as np
import pytest
from lane_dataset.geometry import LaneReference,EvidenceFrame
from lane_dataset.observed_corridor import reconstruct,METHOD
from lane_dataset.admission import interpolation_evidence


def fixture():
    s=np.arange(41,dtype=float);theta=s/40.
    center=np.c_[40*np.sin(theta),40*(1-np.cos(theta)),np.zeros(len(s))]
    normal=np.c_[-np.sin(theta),np.cos(theta),np.zeros(len(s))]
    left=center-1.8*normal;right=center+1.8*normal
    masks=np.zeros((2,len(s)),bool);masks[0,::4]=True;masks[1,2::4]=True
    refs=tuple((EvidenceFrame(10,.5,'wide'),) if value else () for value in masks.ravel())
    support=SimpleNamespace(supported=masks.ravel(),evidence_frames=refs)
    return s,left,right,support


def test_staggered_dashes_use_both_sides_without_claiming_hidden_pixels():
    s,left,right,support=fixture();before=support.supported.copy()
    ref=LaneReference(left,right,s);out=reconstruct(ref,support,[12.,2.,0.],.5)
    assert not np.any(before[:len(s)]&before[len(s):])
    assert len(out.source_indices)>10
    np.testing.assert_array_equal(out.source_point_indices,np.flatnonzero(before))
    np.testing.assert_array_equal(support.supported,before)
    result=interpolation_evidence(dict(arc_m=s.tolist(),left_xyz=left.tolist(),right_xyz=right.tolist()),support,[12.,2.,0.],method=METHOD)
    assert all(result['interval_geometry_valid'])
    assert max(result['interval_geometry_disagreement_m'])<.01
    assert result['original_road_arc_m'][0]==2. and result['original_road_arc_m'][-1]==38.


def test_hidden_side_coordinates_and_map_arc_cannot_change_controller_reference():
    s,left,right,support=fixture();n=len(s)
    original=reconstruct(LaneReference(left,right,s),support,[12.,2.,0.],.5)
    changed_left=left.copy();changed_right=right.copy()
    changed_left[~support.supported[:n],1]+=.15
    changed_right[~support.supported[n:],1]-=.15
    changed=reconstruct(LaneReference(changed_left,changed_right,s*1.7),support,[12.,2.,0.],.5)
    np.testing.assert_array_equal(changed.left_xyz,original.left_xyz)
    np.testing.assert_array_equal(changed.right_xyz,original.right_xyz)
    np.testing.assert_array_equal(changed.s,original.s)
    qa=interpolation_evidence(dict(arc_m=s.tolist(),left_xyz=changed_left.tolist(),right_xyz=changed_right.tolist()),support,[12.,2.,0.],method=METHOD)
    assert not all(qa['interval_geometry_valid'])


def test_missing_side_or_large_gap_cannot_be_bridged():
    s,left,right,support=fixture();n=len(s)
    support.supported[n:]=False
    with pytest.raises(ValueError,match='insufficient_independent'):
        reconstruct(LaneReference(left,right,s),support,[12.,2.,0.],.5)
    s,left,right,support=fixture()
    with pytest.raises(ValueError,match='insufficient_independent'):
        reconstruct(LaneReference(left,right,s),support,[12.,2.,0.],.5,maximum_gap_m=3.)


def test_future_evidence_cannot_support_independent_splines():
    s,left,right,support=fixture()
    with pytest.raises(ValueError,match='future'):
        reconstruct(LaneReference(left,right,s),support,[12.,2.,0.],.4)
