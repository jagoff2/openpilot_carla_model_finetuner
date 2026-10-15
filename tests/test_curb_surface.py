from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

from lane_behavior.curb_surface import detect_curb_step,curb_survey_owner,survey_owned_world
from lane_dataset.geometry import LaneReference


def profile(grade=.02):
    x=np.arange(-.9,.4001,.005)
    z=.03366+grade*x+np.clip(5.4*(x+.195),0,.15)
    return x,z


def test_toe_intersects_measured_road_grade_not_top_or_material_seam():
    x,z=profile()
    report=detect_curb_step(x,z)
    assert report['passed'],report
    assert report['toe_offset_m']==pytest.approx(-.195,abs=1e-6)
    assert report['toe_height_m']==pytest.approx(.03366-.02*.195,abs=1e-6)
    assert report['curb_height_m']==pytest.approx(.15,abs=1e-6)
    assert report['top_offset_m']>report['toe_offset_m']
    assert report['toe_uncertainty_m']<=.005001


@pytest.mark.parametrize('kind',['flat_material_seam','shadow','bump','two_steps','shallow_ramp'])
def test_noncurb_surfaces_do_not_qualify(kind):
    x,z=profile();ground=.03366+.02*x
    if kind in ('flat_material_seam','shadow'):z=ground
    elif kind=='bump':z=ground+.15*np.maximum(0,1-abs(x+.2)/.1)
    elif kind=='two_steps':z=ground+np.clip(5*(x+.3),0,.075)+np.clip(5*(x+.1),0,.075)
    else:z=ground+np.clip(.5*(x+.3),0,.15)
    assert not detect_curb_step(x,z)['passed']


def test_missing_or_nonfinite_collision_samples_reject():
    x,z=profile();z[5]=np.nan
    with pytest.raises(ValueError,match='measured cross-section'):detect_curb_step(x,z)


def test_native_style_top_bevel_is_separate_from_toe_and_keeps_five_mm_limit():
    x=np.arange(-.9,.4001,.005);toe=-.195;join=toe+.13/5.4
    rise=np.where(x<=join,np.maximum(5.4*(x-toe),0.),np.minimum(.13+(x-join),.15))
    result=detect_curb_step(x,.03366+.02*x+rise)
    assert result['passed'] and result['surface_model']=='road_face_top_bevel_plateau'
    assert result['toe_offset_m']==pytest.approx(toe,abs=1e-6)
    assert result['maximum_residual_m']<.005


def test_actual_saved_native_collision_toes_replay_without_using_rendered_depth():
    path=Path(r'G:\lane-centering-dataset-v1\reports\unmarked\surveys\20261015T123950-057a8ba5.json')
    if not path.exists():pytest.skip('immutable native collision survey fixture is not installed')
    report=json.loads(path.read_text())
    rows=[row for row in report['rows'] if row['route_s']==100.]
    assert len(rows)==2
    for row in rows:
        result=detect_curb_step([s['offset_m'] for s in row['samples']],[s['xyz'][2] for s in row['samples']])
        assert result['passed'],result
        assert -.20<result['toe_offset_m']<-.18
        assert result['maximum_residual_m']<.005 and result['toe_uncertainty_m']<.02


class FakeWorld:
    def __init__(self):self.calls=0
    def get_actors(self):return []
    def cast_ray(self,a,b):
        self.calls+=1
        height=.03366+.005*a.x+.003*a.y+np.clip(5.4*(abs(a.y)-4.44),0,.15)
        return [SimpleNamespace(label=25,location=SimpleNamespace(x=a.x,y=a.y,z=height))]


def fake_carla():
    return SimpleNamespace(Location=lambda **kw:SimpleNamespace(**kw),CityObjectLabel=SimpleNamespace(
        Roads=1,RoadLines=24,Ground=25,Sidewalks=2,Terrain=10))


def reference():
    x=np.arange(0.,21.)
    return LaneReference(np.c_[x,x*0-4.635,x*0+.03366],np.c_[x,x*0+4.635,x*0+.03366],x)


def test_owned_coarse_fine_survey_preserves_grade_and_measured_toes(tmp_path):
    world=FakeWorld()
    with curb_survey_owner(9250,tmp_path) as lease:
        report=survey_owned_world(world,reference(),ownership=lease,stations=[5.,7.],
            carla_module=fake_carla(),source_bindings={'fixture':'unit-test'})
    assert report['complete'] and report['passed'],report
    assert report['ray_count']<150
    for row in report['rows']:
        x,y,z=row['toe_xyz']
        assert abs(y)==pytest.approx(4.44,abs=.001)
        assert z==pytest.approx(.03366+.005*x+.003*y,abs=.001)
        assert row['samples'] and row['maximum_residual_m']<.005
    assert not report['training_ready']


def test_no_world_call_without_active_lease_and_budget_never_fills_missing_geometry(tmp_path):
    world=FakeWorld()
    with pytest.raises(ValueError,match='active curb_survey_owner'):
        survey_owned_world(world,reference(),ownership=None,carla_module=fake_carla())
    assert world.calls==0
    with curb_survey_owner(9250,tmp_path) as lease:
        report=survey_owned_world(world,reference(),ownership=lease,stations=[5.,7.],maximum_rays=1,carla_module=fake_carla())
    assert report['budget_exhausted'] and not report['complete'] and not report['passed']
    assert len(report['rows'])==1 and len(report['rows'][0]['samples'])==1
    with pytest.raises(ValueError,match='active curb_survey_owner'):
        survey_owned_world(world,reference(),ownership=lease,carla_module=fake_carla())
