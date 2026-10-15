"""Explicit observed-space permission never changes ordinary right-side driving."""
from dataclasses import replace
import numpy as np
import pytest

from lane_dataset.geometry import EvidenceFrame,LaneReference,ObservableReference
from lane_behavior.passing import (ObservedPassingSpace,PlannedPassingSpace,reference_fingerprint,
    permitted_edges,permission_issue)
from lane_behavior.target import (ObservedRoad,ObservedObstacle,EgoFootprint,BehaviorConfig,
    build_behavior_target,_body,_geometry)
from lane_behavior.planned import PlannedObstacle,planned_scene_geometry
from lane_behavior.window import build_behavior_window
from lane_behavior.teacher import PhysicalTeacher
from lane_behavior.teacher_quality import qualify_teacher_rollout
from test_behavior_teacher import configuration,state
from test_behavior_teacher_dynamics import measured_calibration,independent_motion


def scene(*,observed=False,width=8.88):
    q=np.arange(0.,121.)
    left=np.c_[q,q*0-width/2,q*0];right=np.c_[q,q*0+width/2,q*0]
    frame=EvidenceFrame(20,1.,'wide')
    ref=ObservableReference(left,right,q,[[frame]]*len(q),1.) if observed else LaneReference(left,right,q)
    ego=EgoFootprint(4.791779518,2.163450002,(.029212622,0.))
    # Measured Audi dimensions, fully on pavement and 15 cm from the curb.
    far=width/2-.15;near=far-1.788678526
    polygon=[[48.147315264,near],[51.852684736,near],[51.852684736,far],[48.147315264,far]]
    actor=(ObservedObstacle('parked',polygon,'parked_vehicle',(frame,),'static') if observed
        else PlannedObstacle('parked',polygon,'parked_vehicle','static'))
    return ref,ego,actor


def permission(reference,actor,*,observed=False,width=1.,intervals=None):
    args=dict(reference_sha256=reference_fingerprint(reference),
        allowed_intervals_m=intervals or ((reference.s[0],reference.s[-1]),),
        opposing_width_m=width,obstacle_ids=(actor.identity,),controlled_scene_sha256='a'*64)
    if observed:
        return ObservedPassingSpace(**args,rendered_free_space_sha256='b'*64,
            evidence_frames=(EvidenceFrame(20,1.,'wide'),))
    return PlannedPassingSpace(**args)


@pytest.fixture(scope='module')
def planned():
    ref,ego,actor=scene();space=permission(ref,actor)
    target=planned_scene_geometry(ref,[actor],ego,passing_space=space,obstacle_tracking_reserve_m=.1)
    assert target.feasible,target.reasons
    return target


def test_fully_on_road_car_cannot_pass_under_original_right_half_contract():
    ref,ego,actor=scene()
    target=planned_scene_geometry(ref,[actor],ego,obstacle_tracking_reserve_m=.1)
    assert not target.feasible
    assert 'passing_space' not in target.document()['source']


def test_scoped_pass_crosses_divider_and_retains_five_metres_nominal_recovery(planned):
    assert planned.config.parked_vehicle_clearance_m==.6
    assert planned.obstacle_tracking_reserve_m==.1
    event=planned.excursions[0]
    query=np.array([event.start_s-1.,(event.plateau_start_s+event.plateau_end_s)/2,event.end_s+5.])
    xy,tangent,left,right,valid=planned.sample(query)
    assert valid.all()
    body=_body(xy,tangent,planned.ego.corners())
    assert body[1,:,1].min()<0.
    assert left[1,1]==pytest.approx(-1.)
    assert left[[0,2],1]==pytest.approx([0.,0.])
    assert xy[[0,2],1]==pytest.approx([2.22,2.22])
    assert planned.diagnostics['minimum_actual_obstacle_clearance_surplus_m']>=.1
    assert event.end_s+5.<planned.s[-1]
    assert abs(event.offset_m)<1.7


def test_controller_and_physical_QA_use_only_event_scoped_permission(planned):
    teacher=PhysicalTeacher(planned,measured_calibration(),config=configuration())
    event=planned.excursions[0]
    center,heading,*_=planned.sample(np.array([(event.plateau_start_s+event.plateau_end_s)/2]))
    command=teacher.step(state(x=center[0,0],y=center[0,1],yaw=np.arctan2(heading[0,1],heading[0,0])))
    assert command.accepted,command.reasons
    teacher=PhysicalTeacher(planned,measured_calibration(),config=configuration())
    assert teacher.step(state(x=10.,y=.7)).reasons==('measured_body_leaves_planned_permitted_road_space',)
    rows=independent_motion(PhysicalTeacher(planned,measured_calibration(),config=configuration()),
        start_s=10.,end_s=event.end_s+12.)
    assert rows[-1]['command'].accepted,rows[-1]['command'].reasons
    qa=qualify_teacher_rollout(target=planned,target_sha256=planned.contract_sha256,
        timestamps_s=[r['t'] for r in rows],positions_xyz=[[r['actor'][0],r['actor'][1],0.] for r in rows],
        yaw_rad=[r['yaw'] for r in rows],velocity_world=[[r['velocity'][0],r['velocity'][1],0.] for r in rows],
        lateral_acceleration_mps2=[r['lateral_acceleration'] for r in rows],collision_count=0)
    assert qa['passed'],qa['reasons']
    assert qa['metrics']['sweep']['minimum_permitted_road_clearance_m']>=.2
    assert qa['metrics']['sweep']['minimum_obstacle_clearance_m']['parked']>=.6
    assert 'minimum_right_half_clearance_m' not in qa['metrics']['sweep']


@pytest.mark.parametrize('change,reason',[
    ('stale','passing_space_stale_evidence'),('future','passing_space_missing_or_future_evidence'),
    ('different_road','passing_space_reference_mismatch'),('wrong_actor','passing_space_designated_obstacle_missing'),
    ('outside','passing_space_outside_observed_reference'),('pedestrian','passing_space_requires_static_parked_vehicle'),
    ('moving','passing_space_requires_static_parked_vehicle')])
def test_observed_permission_requires_exact_supported_static_scene(change,reason):
    ref,ego,actor=scene(observed=True);space=permission(ref,actor,observed=True)
    if change=='stale':space=replace(space,evidence_frames=(EvidenceFrame(0,-5.,'wide'),))
    elif change=='future':space=replace(space,evidence_frames=(EvidenceFrame(30,2.,'wide'),))
    elif change=='different_road':space=replace(space,reference_sha256='c'*64)
    elif change=='wrong_actor':space=replace(space,obstacle_ids=('different',))
    elif change=='outside':space=replace(space,allowed_intervals_m=((-1.,120.),))
    elif change=='pedestrian':actor=replace(actor,kind='pedestrian')
    elif change=='moving':actor=replace(actor,motion='moving')
    target=build_behavior_target(ObservedRoad(ref),[actor],ego,passing_space=space)
    assert not target.moving_supervision and reason in target.reasons


def test_partial_or_too_narrow_visible_permission_cannot_cover_a_complete_pass():
    ref,ego,actor=scene(observed=True)
    for space in (permission(ref,actor,observed=True,width=.1),
            permission(ref,actor,observed=True,intervals=((20.,49.),))):
        target=build_behavior_target(ObservedRoad(ref),[actor],ego,passing_space=space)
        assert not target.moving_supervision


def test_observed_future_permission_does_not_create_hidden_origin_geometry():
    ref,ego,actor=scene(observed=True)
    keep=ref.s>=18.;ref=ObservableReference(ref.left_xyz[keep],ref.right_xyz[keep],ref.s[keep],
        [r for r,k in zip(ref.evidence_frames,keep) if k],ref.decision_time_s)
    target=build_behavior_window(ObservedRoad(ref),[actor],ego,
        passing_space=permission(ref,actor,observed=True))
    assert target.moving_supervision,target.reasons
    assert target.source_ego_s is None and target.s[0]>18.
    assert not target.sample(np.array([0.]))[-1][0]
    event=target.excursions[0]
    assert target.sample(np.array([event.plateau_start_s+1.]))[2][0,1]<0


def test_explicit_permission_cannot_change_clear_road_nominal_or_survive_tampering(planned):
    event=planned.excursions[0]
    q=np.array([event.start_s-1.,event.end_s+1.])
    assert planned.sample(q)[0][:,1]==pytest.approx([2.22,2.22])
    assert permitted_edges(planned.reference,q,(),planned.passing_space)[0][:,1]==pytest.approx([0.,0.])
    with pytest.raises(ValueError,match='changed after'):
        replace(planned,passing_space=replace(planned.passing_space,opposing_width_m=2.)).verify_contract()


def test_permission_documents_round_trip_with_distinct_provenance():
    ref,_,actor=scene();planned=permission(ref,actor)
    assert PlannedPassingSpace.from_document(planned.document())==planned
    ref,_,actor=scene(observed=True);observed=permission(ref,actor,observed=True)
    assert ObservedPassingSpace.from_document(observed.document())==observed
    assert permission_issue(planned,ref,[actor],observed=True)=='passing_space_provenance_mismatch'


def test_serialized_teacher_retains_permission_and_distinguishes_the_virtual_divider(planned):
    document=planned.document()
    assert document['source']['passing_space']==planned.passing_space.document()
    profile=document['planned_profile']
    assert np.allclose(np.asarray(profile['virtual_divider_xy'])[:,1],0.)
    assert np.asarray(profile['permitted_left_boundary_xy'])[:,1].min()==pytest.approx(-1.)
    from lane_behavior.admission import reconstruct_teacher
    assert reconstruct_teacher(document).contract_sha256==planned.contract_sha256


def test_observed_episode_QA_replays_the_explicit_permission():
    from lane_behavior.quality import qualify_episode
    from test_behavior_quality import telemetry,visual
    ref,ego,actor=scene(observed=True)
    target=build_behavior_target(ObservedRoad(ref),[actor],ego,passing_space=permission(ref,actor,observed=True))
    assert target.moving_supervision,target.reasons
    report=qualify_episode(**telemetry(target,start=10.,end=85.,step=.125),target=target,
        obstacles=[actor],collision_count=0,visual_evidence=visual([actor]))
    assert report['physical_passed'],report['reasons']
    assert 'minimum_permitted_road_clearance_m' in report['metrics']['sweep']
