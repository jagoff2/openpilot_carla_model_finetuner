"""Observed permission through portable records and actual differentiable loss.

Synthetic visual-proof bindings isolate this integration contract. Rendered
proof generation and inventory replay have separate free-space tests.
"""
from dataclasses import replace
import numpy as np
import pytest
import torch

from lane_behavior.records import build_record,training_corridor
from lane_behavior.loss import behavior_loss
from lane_behavior.target import ObservedRoad,ObservedObstacle
from lane_dataset.geometry import ObservableReference
from test_behavior_passing import scene,permission


def fixture(*,angle=0.,translation=(0.,0.),space=True,bindings=None):
    ref,ego,actor=scene(observed=True)
    if angle or translation!=(0.,0.):
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        shift=np.asarray(translation);left=ref.left_xyz.copy();right=ref.right_xyz.copy()
        left[:,:2]=left[:,:2]@rotation.T+shift;right[:,:2]=right[:,:2]@rotation.T+shift
        ref=ObservableReference(left,right,ref.s,ref.evidence_frames,ref.decision_time_s)
        actor=replace(actor,polygon_xy=np.asarray(actor.polygon_xy)@rotation.T+shift)
        position=np.r_[np.array([30.,2.22])@rotation.T+shift,0.]
    else:position=np.array([30.,2.22,0.])
    granted=permission(ref,actor,observed=True) if space else None
    bound={'road-proof.json':'c'*64,'obstacle-proof.json':'d'*64}
    if space:bound.update(controlled_scene=granted.controlled_scene_sha256,rendered_free_space=granted.rendered_free_space_sha256)
    if bindings is not None:bound=bindings
    record=build_record(ObservedRoad(ref),[actor],ego,label_frame_id=20,label_timestamp_s=1.,
        evidence_bindings=bound,ego_position_world=position,ego_yaw_rad=angle,
        target_mode='observed_future_window',passing_space=granted)
    return record,bound


def converted(**kwargs):
    record,bindings=fixture(**kwargs)
    corridor=training_corridor(record,verified_evidence_bindings=bindings)
    assert corridor is not None,record['reasons']
    return record,corridor


def test_fully_road_parked_pass_supervision_beats_collision_excess_crossing_and_no_return():
    record,corridor=converted()
    distance=np.linspace(0.,50.,201)
    reference=torch.tensor(np.c_[distance,distance*0][None],dtype=torch.float32)
    good=torch.tensor(corridor.sample(corridor.ego_s+distance)[0][None],dtype=torch.float32)
    loss,report=behavior_loss(good,reference,[corridor])
    assert report['valid_labels']==1 and report['obstacle_loss']<1e-5
    # Fully on-road parked Audi requires the actual Tesla body across divider.
    assert float(good[:,:,1].min())+2.22-corridor.width/2<0.
    assert float(good[0,-1,1])==pytest.approx(0.,abs=1e-5)
    nominal=reference.clone().requires_grad_(True)
    collision,collision_report=behavior_loss(nominal,reference,[corridor])
    assert collision>loss+.1 and collision_report['obstacle_loss']>.01
    collision.backward();assert torch.isfinite(nominal.grad).all() and nominal.grad.abs().sum()>0
    # A descent step moves left away from the right-side obstacle.
    assert nominal.grad[0,60:100,1].sum()>0
    wander=good.clone();wander[0,45:135,1]-=.8
    no_return=good.clone();no_return[0,120:,1]=float(good[:,:,1].min())
    assert behavior_loss(wander,reference,[corridor])[0]>loss+.1
    assert behavior_loss(no_return,reference,[corridor])[0]>loss+.1
    assert not record['training_ready']


def test_permission_survives_exact_world_to_ego_record_replay():
    first,a=converted();moved,b=converted(angle=.53,translation=(96.,-41.))
    assert first['record_sha256']!=moved['record_sha256']
    q=np.linspace(0.,50.,151)
    for x,y in zip(a.sample(a.ego_s+q),b.sample(b.ego_s+q)):
        np.testing.assert_allclose(x,y,atol=1e-6)
    assert a.target.passing_space.rendered_free_space_sha256==b.target.passing_space.rendered_free_space_sha256


def test_missing_permission_keeps_old_right_half_contract_and_unbound_permission_rejects():
    record,bindings=fixture(space=False)
    assert not record['moving_supervision']
    assert training_corridor(record,verified_evidence_bindings=bindings) is None
    with pytest.raises(ValueError,match='independently named'):
        fixture(bindings={'road-proof.json':'c'*64})
    record,bindings=fixture()
    changed=dict(bindings,rendered_free_space='e'*64)
    with pytest.raises(ValueError,match='binding mismatch'):
        training_corridor(record,verified_evidence_bindings=changed)
