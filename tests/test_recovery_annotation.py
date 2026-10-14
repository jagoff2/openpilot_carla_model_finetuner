import copy
import numpy as np
from reports.recovery_annotation import recovery_annotation


def fixture():
    arc=np.arange(31,dtype=float)
    left=np.column_stack([arc,np.ones(31)*-2,np.zeros(31)])
    right=left.copy();right[:,1]=2
    geometry=dict(arc_m=arc,left_xyz=left,right_xyz=right,curvature_1pm=np.zeros(31),junction_ranges_m=[])
    manifest=dict(family='constant_radius',maneuver_phase_ranges=dict(recovery=[10.,20.]),terminal_suffix_policy=dict(stop_route_m=30.))
    telemetry=dict(route_s=arc,command_accepted=np.ones(31,bool),position_world=(left+right)*.5)
    return manifest,geometry,telemetry


def test_annotation_uses_recorded_straight_tail_and_preserves_original_proposal():
    m,g,t=fixture();original=copy.deepcopy(m)
    result=recovery_annotation(m,g,t)
    assert result['annotated_phase_ranges']['recovery']==[10.,25.]
    assert result['original_phase_ranges']['recovery']==[10.,20.]
    assert result['uses_label_acceptance_results'] is False
    assert m==original


def test_annotation_refuses_braking_junction_curvature_and_offcenter_tail():
    for kind in ('braking','junction','curve','offset'):
        m,g,t=fixture()
        if kind=='braking':t['command_accepted'][22]=False
        if kind=='junction':g['junction_ranges_m']=[[23.,26.]]
        if kind=='curve':g['curvature_1pm'][22]=.02
        if kind=='offset':t['position_world'][22,1]=.3
        assert recovery_annotation(m,g,t) is None
