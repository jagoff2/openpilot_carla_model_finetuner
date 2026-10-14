from types import SimpleNamespace
import sys
import pytest
from lane_dataset.road_surface import survey_road_surface


def test_independent_collision_survey_preserves_horizontal_geometry(monkeypatch):
    labels=SimpleNamespace(Roads=1,RoadLines=24)
    monkeypatch.setitem(sys.modules,'carla',SimpleNamespace(CityObjectLabel=labels,Location=SimpleNamespace))
    calls=[]
    def cast(start,end):
        calls.append((start,end))
        return [SimpleNamespace(label=1,location=SimpleNamespace(x=start.x,y=start.y,z=.22))]
    route=dict(left_xyz=[[10.,-2.,0.]],right_xyz=[[10.,2.,0.]],center_xyz=[[10.,0.,0.]],arc_m=[0.])
    result,report=survey_road_surface(SimpleNamespace(cast_ray=cast),route)
    assert result['left_xyz']==[[10.,-2.,.22]]
    assert result['right_xyz']==[[10.,2.,.22]]
    assert result['arc_m']==route['arc_m']
    assert route['left_xyz']==[[10.,-2.,0.]]
    assert report['maximum_absolute_vertical_correction_m']==pytest.approx(.22)
    assert len(calls)==7
    with pytest.raises(ValueError,match='no physical road surface'):
        survey_road_surface(SimpleNamespace(cast_ray=lambda a,b:[]),route)
