import numpy as np
import pytest

from lane_dataset.maneuvers import propose_maneuver, route_desire


def route_fixture():
    arc=np.arange(101,dtype=float)
    return dict(arc_m=arc.tolist(),curvature_1pm=np.where((arc>=30)&(arc<=70),.02,0).tolist(),
                junction_ranges_m=[[30.,70.]])


def test_turn_intent_follows_physical_route_progress_instead_of_fixed_frame():
    task=dict(family="intersection_turn",direction="left",desire_route_range_m=[20.,70.])
    assert route_desire(task,90,5.)[0]==1.
    assert route_desire(task,300,25.)[1]==1.
    assert route_desire(task,500,75.)[0]==1.
    assert sum(route_desire(task,300,25.))==1.
    with pytest.raises(ValueError,match="spatial bounds"):
        route_desire({**task,"desire_route_range_m":[70.,20.]},100,30.)


def test_proposed_phases_cover_actual_bend_and_remain_pending_review():
    result=propose_maneuver(route_fixture(),"intersection_turn")
    assert result["annotation_status"]=="proposal_requires_rendered_review"
    assert result["annotation_blockers"]==[]
    assert result["desire_route_range_m"]==[20.,70.]
    assert result["intent_required_route_range_m"]==[25.,31.]
    assert result["maneuver_phase_ranges"]["approach"]==[20.,30.]
    assert result["maneuver_phase_ranges"]["recovery"]==[70.,80.]
    assert set(result["maneuver_phase_ranges"])=={"approach","entry","apex","exit","recovery"}


def test_straight_driving_cannot_fabricate_a_recovery_event():
    result=propose_maneuver(route_fixture(),"straight_recovery")
    assert result["annotation_blockers"]
    assert "maneuver_phase_ranges" not in result


def test_missing_junction_geometry_cannot_assign_branch_scope():
    route=route_fixture()
    route["junction_ranges_m"]=[]
    result=propose_maneuver(route,"intersection_turn")
    assert "intersection_road_scope_missing" in result["annotation_blockers"]
    assert "intent_required_route_range_m" not in result
