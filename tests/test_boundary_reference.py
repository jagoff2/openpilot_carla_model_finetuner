from copy import deepcopy
import numpy as np
import pytest
from lane_dataset.boundary_reference import lane_facing_marking_reference


def test_double_marking_uses_lane_facing_stripe_without_moving_single_line():
    route=dict(left_xyz=[[0.,-1.6,0.],[1.,-1.6,0.]],right_xyz=[[0.,1.6,0.],[1.,1.6,0.]],center_xyz=[[0.,0.,0.],[1.,0.,0.]],arc_m=[0.,1.])
    markings=[dict(left=dict(type='SolidSolid',width_m=.125),right=dict(type='Solid',width_m=.125)),
              dict(left=dict(type='Broken',width_m=.125),right=dict(type='SolidBroken',width_m=.125))]
    result=lane_facing_marking_reference(route,markings)
    np.testing.assert_allclose(result['left_xyz'][0],[0.,-1.475,0.])
    np.testing.assert_allclose(result['right_xyz'][0],route['right_xyz'][0])
    np.testing.assert_allclose(result['right_xyz'][1],[1.,1.475,0.])
    assert result['nominal_left_xyz']==route['left_xyz']
    assert route['left_xyz'][0][1]==-1.6
    bad=deepcopy(markings);bad[0]['left']['width_m']=.8
    with pytest.raises(ValueError,match='30 cm'):lane_facing_marking_reference(route,bad)
    with pytest.raises(ValueError,match='cross-section'):lane_facing_marking_reference(route,[])
