import math
from dataclasses import replace

import numpy as np
import pytest

from lane_dataset.controller import (BicycleMPC, MPCConfig, SteeringCalibration,
                                     VehicleState, fit_steering_calibration,
                                     fit_rate_limited_calibration, _advance_steering,
                                     reference_lateral_acceleration)
from lane_dataset.geometry import (EvidenceFrame, LaneReference, carla_to_device_points,
                                   recovery_reference, single_valued_forward_prefix)
from lane_dataset.quality import qualify_episode, swept_body_clearance


def road(curvature=0., length=100., width=3.6, bank=0., count=301):
    s=np.linspace(0,length,count)
    if curvature:
        theta=s*curvature
        center=np.c_[np.sin(theta)/curvature,(1-np.cos(theta))/curvature,np.zeros(count)]
    else:
        theta=np.zeros(count);center=np.c_[s,np.zeros(count),np.zeros(count)]
    normal=np.c_[-np.sin(theta),np.cos(theta),np.full(count,math.tan(bank))]
    return LaneReference.from_boundaries(center-normal*width/2,center+normal*width/2,s)


def observed(reference,t=0.,mask=None):
    mask=np.ones(len(reference.s),bool) if mask is None else mask
    return reference.admit(mask,[(EvidenceFrame(12,t,"wide"),)]*len(mask),t)


@pytest.mark.parametrize("curvature",[0.,.025,-.025,.08,-.08])
def test_geometry_and_steering_sign(curvature):
    ref=road(curvature,length=40)
    _,_,k,_=ref.sample(np.linspace(2,38,40))
    assert np.max(np.abs(k-curvature))<2e-6
    mpc=BicycleMPC(SteeringCalibration.fixture(),MPCConfig(require_calibration=False))
    command=mpc.control(VehicleState(0,0,0,4,4*curvature),observed(ref),4)
    assert command.accepted,command
    assert np.sign(command.curvature_1pm)==np.sign(curvature) or abs(command.curvature_1pm)<1e-6
    assert abs(command.curvature_1pm-curvature)<.002
    assert abs(command.carla_steer)<=1
    assert command.min_constraint_margin>=-1e-4


def test_initial_offset_target_does_not_move_observed_lane_boundaries():
    reference=observed(road());left=reference.left_xyz.copy();right=reference.right_xyz.copy()
    mpc=BicycleMPC(SteeringCalibration.fixture(),MPCConfig(require_calibration=False))
    command=mpc.control(VehicleState(5.,.25,0.,3.5,0.),reference,3.5,desired_offset_m=.25)
    assert command.accepted and command.constraint_margins['body_clearance']>0
    np.testing.assert_array_equal(reference.left_xyz,left);np.testing.assert_array_equal(reference.right_xyz,right)
    with pytest.raises(ValueError,match='initial offset'):
        mpc.control(VehicleState(5.,.25,0.,3.5,0.),reference,3.5,desired_offset_m=.4)


def test_hidden_geometry_cannot_influence_admitted_derivatives_or_command():
    ref=road(.01)
    left,right=ref.left_xyz.copy(),ref.right_xyz.copy()
    left[100:,1]+=8;right[100:,1]+=8
    altered=LaneReference(left,right,ref.s)
    mask=np.arange(len(ref.s))<95
    a,b=observed(ref,mask=mask),observed(altered,mask=mask)
    assert np.array_equal(a.sample(np.arange(1,30))[2],b.sample(np.arange(1,30))[2])
    cfg=MPCConfig(require_calibration=False)
    state=VehicleState(0,0,0,4,.04)
    ca=BicycleMPC(SteeringCalibration.fixture(),cfg).control(state,a,4)
    cb=BicycleMPC(SteeringCalibration.fixture(),cfg).control(state,b,4)
    assert ca.accepted and cb.accepted
    assert ca.curvature_1pm==cb.curvature_1pm
    with pytest.raises(TypeError,match="full map"):
        BicycleMPC(SteeringCalibration.fixture(),cfg).control(state,ref,4)


def test_observation_rejects_future_evidence_and_unsupported_holes():
    ref=road()
    refs=[(EvidenceFrame(42,1.),)]*len(ref.s)
    with pytest.raises(ValueError,match="future"):
        ref.admit(np.ones(len(ref.s),bool),refs,0.)
    mask=np.ones(len(ref.s),bool);mask[2]=False
    with pytest.raises(ValueError,match="insufficient"):
        ref.admit(mask,refs,1.)


def test_offcenter_recovery_begins_at_recorded_pose():
    ref=observed(road())
    t,p=recovery_reference(ref,[2.,.45],.03,5.,.02)
    assert np.array_equal(p[0,:2],[2.,.45])
    assert abs(p[-1,1])<1e-10
    initial_yaw=math.atan2(p[1,1]-p[0,1],p[1,0]-p[0,0])
    assert abs(initial_yaw-.03)<.002
    mpc=BicycleMPC(SteeringCalibration.fixture(),MPCConfig(require_calibration=False))
    command=mpc.control(VehicleState(2.,.45,.03,5.,.02),ref,5.)
    assert command.accepted,command.reasons
    assert np.array_equal(command.predicted_states[0,:4],[2.,.45,.03,5.])
    assert abs(command.predicted_states[-1,1])<.10


def test_first_visible_goal_ahead_does_not_shrink_stopping_distance():
    ref=road(length=20.,count=81)
    observed_ahead=ref.admit(np.ones(81,bool),[(EvidenceFrame(1,0.,"wide"),)]*81,0.,start_index=32)
    command=BicycleMPC(SteeringCalibration.fixture(),MPCConfig(require_calibration=False)).control(
        VehicleState(0,0,0,5,0),observed_ahead,5.)
    assert command.accepted,command.reasons
    assert command.target_speed_mps==5.
    assert command.curvature_1pm==pytest.approx(0.,abs=1e-6)


def test_distant_paint_transition_does_not_step_cruise_speed():
    reference=road(length=60.,count=121)
    left=reference.left_xyz.copy();right=reference.right_xyz.copy()
    # A double stripe ending moves the chosen paint center by 12.5 cm.
    # It is well beyond this decision's driven horizon.
    left[reference.s>=40.,1]+=.125
    corridor=observed(LaneReference(left,right,reference.s))
    config=MPCConfig(require_calibration=False)
    controller=BicycleMPC(SteeringCalibration.fixture(),config)
    command=controller.control(VehicleState(5.,0.,0.,5.,0.),corridor,5.)
    assert command.accepted,command.reasons
    assert command.target_speed_mps==5.
    dense=controller._rollout(VehicleState(5.,0.,0.,5.,0.),controller._previous_solution,dense=True)
    ay=reference_lateral_acceleration(dense[:,3],dense[:,4],config.control_dt_s,0.,0.)
    assert np.max(np.abs(ay))<=config.max_lateral_acceleration_mps2+1e-4
    assert np.max(np.abs(np.diff(ay)/config.control_dt_s))<=config.max_lateral_jerk_mps3+1e-4


@pytest.mark.parametrize('retry_converges',[True,False])
def test_iteration_limit_requires_a_converged_retry(monkeypatch,retry_converges):
    import lane_dataset.controller as module
    original=module.minimize;calls=[]
    def limited(*args,**kwargs):
        result=original(*args,**kwargs);calls.append(result.success)
        assert result.success
        if len(calls)==1 or not retry_converges:
            result.success=False;result.status=9;result.message='Iteration limit reached'
        return result
    monkeypatch.setattr(module,'minimize',limited)
    command=BicycleMPC(SteeringCalibration.fixture(),MPCConfig(require_calibration=False)).control(
        VehicleState(5,0,0,5,0),observed(road()),5.)
    assert len(calls)==2
    assert command.accepted is retry_converges
    assert command.optimizer_success is retry_converges
    assert command.min_constraint_margin>=-1e-4


def test_prediction_horizon_preserves_observed_body_and_stopping_distance():
    controller=BicycleMPC(SteeringCalibration.fixture(),MPCConfig(require_calibration=False))
    command=controller.control(VehicleState(5.,0.,0.,3.5,0.),observed(road(length=12.5,count=51)),3.5)
    assert command.accepted,command.reasons
    assert 4<len(command.predicted_states)<17
    assert command.predicted_states[-1,0]+controller.config.vehicle_length_m/2<=12.5+1e-4
    expanded=controller.control(VehicleState(5.175,0.,0.,3.5,0.,timestamp_s=.05),observed(road(),t=.05),3.5)
    assert expanded.accepted and len(expanded.predicted_states)==17
    with pytest.raises(ValueError,match='stopping_corridor'):
        BicycleMPC(SteeringCalibration.fixture(),MPCConfig(require_calibration=False)).control(
            VehicleState(5.,0.,0.,3.5,0.),observed(road(length=10.,count=41)),3.5)


def test_executed_command_sequence_limits_actor_reference_jerk_under_goal_jitter():
    calibration=replace(SteeringCalibration.fixture(delay_s=0.,time_constant_s=.023),reference_offset_m=1.6)
    controller=BicycleMPC(calibration,MPCConfig(require_calibration=False))
    angles=[0.,0.];v=3.5;dt=.05
    for i in range(10):
        ref=road();left=ref.left_xyz.copy();right=ref.right_xyz.copy()
        left[:,1]+=.025*(-1)**i;right[:,1]+=.025*(-1)**i
        command=controller.control(VehicleState(5+i*v*dt,0.,0.,v,0.,timestamp_s=i*dt,lateral_acceleration_mps2=0.),
            observed(LaneReference(left,right,ref.s),t=i*dt),v)
        assert command.accepted,command.reasons
        angles.append(command.steering_angle_rad)
        rates=v*np.tan(angles[-3:])/calibration.wheelbase_m
        accelerations=v*rates[1:]+calibration.reference_offset_m*np.diff(rates)/dt
        assert abs(np.diff(accelerations)[0]/dt)<=controller.config.max_executed_command_jerk_mps3+1e-4


def test_hairpin_is_time_parametric_and_fixed_forward_lane_stops():
    ref=road(.08,length=math.pi/.08)
    points=carla_to_device_points(ref.center_xyz,[0,0,0],0)
    prefix=single_valued_forward_prefix(points)
    assert 140<prefix.sum()<160
    assert points[-1,1]>20
    assert not prefix[-1]


def synthetic_calibration():
    dt=.05;n=480;t=np.arange(n)*dt
    u=np.array(([0.]*20+[.1]*20+[0.]*20+[-.1]*20)*6)
    v=5.+.3*np.sin(t*.2);gain=.58;tau=.18;delay=3
    delta=np.zeros(n)
    for i in range(1,n):
        delta[i]=delta[i-1]+(1-math.exp(-dt/tau))*(gain*u[max(0,i-1-delay)]-delta[i-1])
    r=v*np.tan(delta)/2.8
    return t,u,v,r


def test_calibration_uses_measured_holdout_and_finds_delay():
    t,u,v,r=synthetic_calibration()
    result=fit_steering_calibration(t,u,v,r,2.8,.65)
    assert result.qualified
    assert result.delay_s==pytest.approx(.15)
    assert result.time_constant_s==pytest.approx(.18,abs=.001)
    assert result.steering_gain_rad==pytest.approx(.58,abs=.001)
    r[-120:]*=1.9
    failed=fit_steering_calibration(t,u,v,r,2.8,.65)
    assert not failed.qualified
    assert failed.steering_gain_rad==pytest.approx(.58,abs=.001)


def test_production_rejects_analytical_calibration():
    with pytest.raises(ValueError,match="held-out"):
        BicycleMPC(SteeringCalibration.fixture())


def test_rate_limited_calibration_and_nonlinear_command_roundtrip():
    commands=[]
    for _ in range(3):
        for amplitude in (.05,.1,.2,.25):
            for value in (0.,amplitude,0.,-amplitude):commands.extend([value]*10)
    u=np.asarray(commands);t=np.arange(len(u))*.05;v=np.full(len(u),5.)
    delta=np.zeros(len(u));motor=0.
    for i in range(1,len(u)):
        target=1.1*u[i-1]-1.2*u[i-1]**3
        motor,delta[i]=_advance_steering(motor,delta[i-1],target,.05,.03,3.7)
    fitted=fit_rate_limited_calibration(t,u,v,v*np.tan(delta)/3.,3.,1.22)
    assert fitted.qualified
    assert fitted.response_rate_rps==pytest.approx(3.7,abs=.02)
    assert fitted.steering_cubic_rad==pytest.approx(-1.2,abs=.02)
    for fraction in np.linspace(-.25,.25,30):
        assert fitted.steer_from_angle(fitted.angle_from_steer(fraction))==pytest.approx(fraction,abs=1e-8)


def test_actor_reference_offset_matches_measured_lateral_velocity():
    calibration=replace(SteeringCalibration.fixture(delay_s=0.),reference_offset_m=1.5)
    config=MPCConfig(require_calibration=False)
    mpc=BicycleMPC(calibration,config);delta=math.atan(2.8*.04)
    initial=VehicleState(0,0,0,5,.2,delta)
    variables=np.r_[np.full(config.horizon_steps,delta),np.zeros(config.horizon_steps)]
    predicted=mpc._rollout(initial,variables,dense=True)
    first_lateral_velocity=(predicted[1,1]-predicted[0,1])/.05
    # Motion over the first tick includes initial lateral velocity l*r=.3
    # plus half a tick of turning acceleration ~.025 m/s.
    assert first_lateral_velocity==pytest.approx(.325,abs=.003)


def test_actor_reference_acceleration_includes_offset_times_yaw_acceleration():
    t=np.arange(21)*.05;speed=np.full(len(t),2.5);rate=.2*t;offset=1.5
    # A yaw-rate ramp carries 0.3 m/s^2 acceleration at the offset point,
    # even when the steady centripetal term is nearly zero.
    expected=np.sqrt(speed**2-(offset*rate)**2)*rate+.3
    actual=reference_lateral_acceleration(speed,rate,.05,offset,.3)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
    rear=reference_lateral_acceleration(speed,rate,.05,0.,0.)
    np.testing.assert_allclose(rear,speed*rate)


def test_recovery_prediction_limits_actor_reference_jerk_from_measured_initial_acceleration():
    calibration=replace(SteeringCalibration.fixture(delay_s=0.,time_constant_s=.03),reference_offset_m=1.5)
    config=MPCConfig(require_calibration=False)
    controller=BicycleMPC(calibration,config)
    initial=VehicleState(2.,.45,.052,2.5,0.,lateral_acceleration_mps2=0.)
    command=controller.control(initial,observed(road(width=3.9)),2.5)
    assert command.accepted,command.reasons
    dense=controller._rollout(initial,controller._previous_solution,dense=True)
    acceleration=reference_lateral_acceleration(dense[:,3],dense[:,4],.05,1.5,0.)
    assert np.max(np.abs(acceleration))<=2.5+1e-4
    assert np.max(np.abs(np.diff(acceleration)/.05))<=4.+1e-3


def test_calibrated_recovery_cannot_plan_braking_outside_identified_speed_domain():
    calibration=replace(SteeringCalibration.fixture(delay_s=0.,time_constant_s=.03),
        reference_offset_m=1.5,qualified=True,speed_min_mps=2.45,speed_max_mps=2.55)
    command=BicycleMPC(calibration).control(VehicleState(2.,.45,.052,2.5,0.,lateral_acceleration_mps2=0.),
        observed(road(width=3.9)),2.5)
    assert command.accepted,command.reasons
    assert command.predicted_states[:,3].min()>=2.45-1e-4
    assert command.predicted_states[:,3].max()<=2.55+1e-4


def test_unmodeled_initial_acceleration_does_not_create_steering_feedback():
    calibration=replace(SteeringCalibration.fixture(delay_s=0.,time_constant_s=.03),reference_offset_m=1.5)
    config=MPCConfig(require_calibration=False)
    state=VehicleState(2.,.02,0.,3.5,0.,lateral_acceleration_mps2=0.)
    ref=observed(road())
    clean=BicycleMPC(calibration,config).control(state,ref,3.5)
    disturbed=BicycleMPC(calibration,config).control(replace(state,lateral_acceleration_mps2=.7),ref,3.5)
    assert clean.accepted and disturbed.accepted
    # A suspension disturbance must not be translated into a steering pulse.
    assert disturbed.carla_steer==pytest.approx(clean.carla_steer,abs=1e-9)


@pytest.mark.parametrize("family",["tightening","s_bend","hairpin"])
def test_procedural_maneuver_mpc_constraints(family):
    from lane_dataset.procedural import parameters_for_split, sample_reference_line
    points=sample_reference_line(parameters_for_split("development",family,1),.25)
    center=points[:,1:].copy();center[:,1]*=-1
    tangent=np.gradient(center[:,:2],points[:,0],axis=0)
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    normal=np.c_[-tangent[:,1],tangent[:,0],np.zeros(len(tangent))]
    ref=LaneReference(center-normal*1.8,center+normal*1.8,points[:,0])
    xyz,yaw,k,_=ref.sample(80.)
    state=VehicleState(xyz[0],xyz[1],float(yaw),3.,float(3*k))
    command=BicycleMPC(SteeringCalibration.fixture(),MPCConfig(require_calibration=False)).control(state,observed(ref),3.)
    assert command.accepted,command.reasons
    assert command.min_constraint_margin>=-1e-4


def quality_inputs():
    t=np.arange(41)*.05
    return dict(timestamps_s=t,positions_xyz=np.c_[5+t*4,np.zeros(len(t)),np.zeros(len(t))],
        yaw_rad=np.zeros(len(t)),speeds_mps=np.full(len(t),4.),lateral_acceleration_mps2=np.zeros(len(t)),
        steering_fractions=np.zeros(len(t)),reference=road(),vehicle_length_m=4.6,vehicle_width_m=1.85,
        collisions=0,rendered_boundary_agreement=True,completed_maneuver=True,intended_progress_m=7.9)


def test_quality_exact_gates_require_rendered_and_collision_evidence():
    data=quality_inputs();report=qualify_episode(**data)
    assert report.accepted,report.reasons
    data["positions_xyz"][:,1]=.11
    rejected=qualify_episode(**data)
    assert "mean_signed_center_offset" in rejected.reasons
    assert "p95_absolute_center_offset" in rejected.reasons
    data=quality_inputs();data["collisions"]=None;data["rendered_boundary_agreement"]=None
    rejected=qualify_episode(**data)
    assert "collision_sensor_evidence_missing" in rejected.reasons
    assert "rendered_boundary_agreement_missing_or_failed" in rejected.reasons


def test_swept_body_checks_heading_and_banked_boundary():
    ref=road(width=3.6,bank=.1)
    straight=swept_body_clearance(ref,[[5,0,0],[6,0,0]],[0,0],4.6,1.85)
    assert straight["minimum_boundary_clearance_m"]>.8
    turn=swept_body_clearance(ref,[[5,0,0],[6,0,0]],[0,.5],4.6,1.85)
    assert turn["minimum_boundary_clearance_m"]<.15


def test_quality_recovery_not_exempt_after_four_seconds():
    data=quality_inputs();data["positions_xyz"][:,1]=.3
    report=qualify_episode(**data,recovery=True)
    assert not report.accepted
    assert "recovery_did_not_converge_within_four_seconds" in report.reasons


def test_recovery_timer_starts_at_measured_release_and_stationary_suffix_uses_body_heading():
    data=quality_inputs();t=np.arange(201)*.05
    speed=np.where(t<8.,4.,0.)
    offset=np.where(t<5.,.25,np.maximum(0.,.25*(1-(t-5.)/2.)))
    data.update(timestamps_s=t,positions_xyz=np.c_[5+4*np.minimum(t,8.),offset,np.zeros(len(t))],
        yaw_rad=np.zeros(len(t)),speeds_mps=speed,lateral_acceleration_mps2=np.zeros(len(t)),
        steering_fractions=np.zeros(len(t)),velocity_world=np.c_[speed,np.zeros(len(t)),np.zeros(len(t))],intended_progress_m=31.9)
    report=qualify_episode(**data,recovery=True,recovery_onset_s=5.)
    assert report.accepted,report.reasons
    assert 1.<report.metrics['recovery_time_s']<2.
    assert 'recovery_did_not_converge_within_four_seconds' in qualify_episode(**data,recovery=True).reasons
