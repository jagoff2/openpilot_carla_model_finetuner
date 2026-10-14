import numpy as np
import pytest

from lane_dataset.geometry import LaneReference
from lane_dataset.quality import swept_body_clearance


def straight():
    x=np.linspace(0,30,61)
    left=np.column_stack([x,np.full_like(x,-2.),np.zeros_like(x)])
    right=left.copy();right[:,1]=2.
    return LaneReference(left,right)


def test_banked_body_uses_roof_height_and_actor_bbox_offset():
    bank=.15
    result=swept_body_clearance(straight(),[[10,0,0],[10,0,0]],[0,0],4.,1.8,
        rotation_world=[[bank,0,0],[bank,0,0]],vehicle_height_m=2.,vehicle_bbox_center_xyz=[0,0,1.])
    expected=2.-(.9*np.cos(bank)+2.*np.sin(bank))
    assert result['minimum_boundary_clearance_m']==pytest.approx(expected,abs=1e-7)
    assert result['body_footprint_mode']=='transformed_3d_bounding_box'
    assert expected < 2.-.9


def test_actor_local_box_displacement_changes_clearance():
    result=swept_body_clearance(straight(),[[10,0,0],[10,0,0]],[0,0],4.,1.8,
        rotation_world=[[0,0,0],[0,0,0]],vehicle_height_m=2.,vehicle_bbox_center_xyz=[0,.3,1.])
    assert result['minimum_boundary_clearance_m']==pytest.approx(.8,abs=1e-7)


def test_roll_sweep_detects_crossing_between_dense_endpoints():
    result=swept_body_clearance(straight(),[[10,0,0],[10,0,0]],[0,0],4.,1.8,
        rotation_world=[[-.8,0,0],[.8,0,0]],vehicle_height_m=4.,vehicle_bbox_center_xyz=[0,0,2.])
    assert result['swept_boundary_crossing']
    assert result['minimum_boundary_clearance_m'] < 0


def circle(radius=11., width=3.3, knot_count=241):
    angle=np.linspace(-1.2,1.2,knot_count)
    center=np.column_stack([radius*np.sin(angle),radius*(1-np.cos(angle)),np.zeros(len(angle))])
    normal=np.column_stack([-np.sin(angle),np.cos(angle),np.zeros(len(angle))])
    return LaneReference(center-width/2*normal,center+width/2*normal,radius*(angle-angle[0]))


def test_hairpin_inner_side_midpoint_fails_clearance_when_all_corners_pass():
    # On the inside of a tight curve, a tangent rectangle's side midpoint is
    # closer to the boundary than either of its corners. The old corner-only
    # test returned 0.41694979 m for this recoverable starting position.
    result=swept_body_clearance(circle(),[[0,.45,0],[0,.45,0]],[0,0],4.792,2.163,
        rotation_world=np.zeros((2,3)),vehicle_height_m=1.5)
    exact_midpoint_clearance=3.3/2-.45-2.163/2
    assert exact_midpoint_clearance==pytest.approx(.1185)
    assert result['minimum_boundary_clearance_m']<=exact_midpoint_clearance+1e-8
    assert result['minimum_boundary_clearance_m']>=exact_midpoint_clearance-.001
    assert result['minimum_boundary_clearance_m']<.15
    assert not result['swept_boundary_crossing']
    assert result['body_clearance_method'].startswith('whole_convex_polygon')


def test_hairpin_edge_interior_crosses_boundary_while_corners_remain_inside():
    result=swept_body_clearance(circle(),[[0,.65,0],[0,.65,0]],[0,0],4.792,2.163,
        rotation_world=np.zeros((2,3)),vehicle_height_m=1.5)
    assert result['swept_boundary_crossing']
    assert result['minimum_boundary_clearance_m']<=0


def test_body_wholly_outside_lane_is_not_mistaken_for_positive_clearance():
    result=swept_body_clearance(straight(),[[10,5,0],[10,5,0]],[0,0],4.,1.8)
    assert result['swept_boundary_crossing']
    assert result['minimum_boundary_clearance_m']<0


def test_boundary_chord_bound_remains_conservative_with_sparse_spline_knots():
    from lane_dataset.quality import QualityConfig
    reference=circle(knot_count=13)
    coarse=swept_body_clearance(reference,[[0,.45,0],[0,.45,0]],[0,0],4.792,2.163,
        QualityConfig(maximum_boundary_chord_spacing_m=2.))
    fine=swept_body_clearance(reference,[[0,.45,0],[0,.45,0]],[0,0],4.792,2.163,
        QualityConfig(maximum_boundary_chord_spacing_m=.02))
    assert coarse['boundary_chord_error_bound_m']>fine['boundary_chord_error_bound_m']
    assert coarse['minimum_boundary_clearance_m']<=fine['minimum_boundary_clearance_m']+1e-7
    assert fine['minimum_boundary_clearance_m']<.15
