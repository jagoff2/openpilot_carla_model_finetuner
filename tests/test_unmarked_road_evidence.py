import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from lane_behavior.evidence import FrameStamp, pack_model_rgb
from lane_behavior.road_evidence import (FrameRoadEvidence, _digest, assemble_observed_road,
    measure_road_frame)
from lane_dataset.config import sha256_file
from lane_dataset.geometry import LaneReference
from lane_dataset.visibility import decode_carla_depth_bgra


def frame(frame_id=10,time=1.,left=(0,2,4,6),right=(0,2,4,6),camera='narrow',xscale=.5):
    stamp=FrameStamp(frame_id,time,camera)
    knots=tuple(dict(side=side,source_index=i,observed_xyz=[i*xscale,-4. if side=='left' else 4.,0.])
        for side,ids in (('left',left),('right',right)) for i in ids)
    audit=dict(stamp=dict(frame_id=frame_id,timestamp_s=time,camera=camera),reference_sha256='ref',knots=list(knots))
    return FrameRoadEvidence(stamp,'ref',knots,audit,_digest(audit))


def assemble(frames,**kw):
    return assemble_observed_road(frames,decision=FrameStamp(10,1.,'narrow'),position_world=[0.,0.,0.],**kw)


def test_common_span_only_and_observed_knots_rebuild_splines():
    result=assemble([frame(left=(0,2,4,6,8),right=(2,4,6,8,10))])
    assert result.road is not None and not result.reasons
    assert result.audit['source_indices']==[2,4,6,8]
    ref=result.road.reference
    np.testing.assert_allclose(ref.left_xyz[:,0],[1,2,3,4])
    np.testing.assert_allclose(.25*ref.left_xyz+.75*ref.right_xyz,
        np.c_[[1,2,3,4],np.full(4,2.),np.zeros(4)])
    assert all(refs for refs in ref.evidence_frames)


def test_causal_history_can_supply_opposite_side_without_future_evidence():
    past=frame(9,.75,left=(0,2,4,6),right=())
    current=frame(left=(),right=(0,2,4,6))
    assert assemble([past,current]).road is not None
    future=frame(11,1.05,left=(0,2,4,6),right=())
    result=assemble([future,current])
    assert result.road is None
    assert result.audit['distinct_observed_knots']=={'left':0,'right':4}
    assert result.audit['excluded_frames'][0]['reason']=='future_or_stale'


def test_stale_evidence_and_unbound_current_frame_cannot_support_target():
    result=assemble([frame(9,-4.01),frame(left=(),right=())])
    assert result.road is None
    assert result.audit['distinct_observed_knots']=={'left':0,'right':0}
    assert assemble([frame(9,.95)]).reasons==('current_decision_frame_not_bound',)


def test_sparse_edges_and_disjoint_spans_are_not_filled_from_map():
    assert assemble([frame(left=(0,2,40,42))]).road is None
    result=assemble([frame(left=(0,2,4,6),right=(20,22,24,26))])
    assert result.reasons==('observed_boundaries_have_no_common_corridor',)


def test_replayed_proof_mutation_and_mixed_source_identities_rejected():
    original=frame();changed=copy.deepcopy(original)
    changed.knots[0]['observed_xyz'][1]=-3.
    with pytest.raises(ValueError,match='changed'):
        assemble([changed])
    other=copy.deepcopy(frame(9,.95))
    other.audit['reference_sha256']='other'
    other=replace(other,reference_sha256='other',proof_sha256=_digest(other.audit))
    with pytest.raises(ValueError,match='mixes'):
        assemble([original,other])


@pytest.mark.parametrize('limits',[dict(maximum_age_s=5.01),dict(maximum_gap_m=8.01)])
def test_temporal_or_spatial_thresholds_cannot_be_relaxed(limits):
    with pytest.raises(ValueError,match='cannot exceed'):
        assemble([frame()],**limits)


@pytest.fixture(scope='module')
def raw_road(tmp_path_factory):
    root=tmp_path_factory.mktemp('raw-road-evidence')
    h,w=1024,2048
    yy,xx=np.indices((h,w));fx=2000.;cy=512.;cx=1024.
    denominator=(yy+.5-cy)/fx
    depth=np.divide(1.3,denominator,out=np.full((h,w),1000.),where=denominator>0)
    depth=np.clip(depth,0.,1000.)
    encoded=np.rint(depth/1000.*(256**3-1)).astype(np.uint32)
    depth_bgra=np.stack(((encoded>>16)&255,(encoded>>8)&255,encoded&255,np.full_like(encoded,255)),axis=-1).astype(np.uint8)
    depth=decode_carla_depth_bgra(depth_bgra).astype(np.float32)
    lateral=(xx+.5-cx)/fx*depth
    road=(abs(lateral)<4)&(denominator>0)
    tags=np.where(road,1,25).astype(np.uint8)
    gray=np.where(road,60,180).astype(np.uint8)
    rgb=np.repeat(gray[...,None],3,axis=-1)
    bgra=np.concatenate((rgb[...,::-1],np.full((h,w,1),255,np.uint8)),axis=-1)
    warp=np.array([[2.,0.,512.],[0.,2.,480.],[0.,0.,1.]])
    data=dict(rgb=rgb,rgb_bgra=bgra,depth_m=depth,depth_bgra=depth_bgra,semantic_camera_ids=tags,
        packed=pack_model_rgb(rgb,warp),intrinsics=np.array([[fx,0,cx],[0,fx,cy],[0,0,1.]]),
        camera_from_world=np.array([[0.,1.,0.,0.],[0.,0.,-1.,1.3],[1.,0.,0.,0.],[0.,0.,0.,1.]]),
        model_to_source=warp)
    payload=root/'frame.npz';np.savez_compressed(payload,**data)
    fixture=root/'depth.json';fixture.write_text(json.dumps({'narrow':dict(qualified=True,depth_mode='forward')}))
    stamp=FrameStamp(10,1.,'narrow');x=np.arange(0.,65.,.5)
    reference=LaneReference(np.c_[x,np.full(len(x),-4.),x*0],np.c_[x,np.full(len(x),4.),x*0],x)
    kwargs=dict(payload=payload,payload_sha256=sha256_file(payload),stamp=stamp,
        modality_stamps={key:stamp for key in ('rgb','depth','packed')},semantic_sensor_stamp=stamp,
        depth_fixture=fixture,depth_fixture_sha256=sha256_file(fixture),reference=reference,current_s=0.)
    return data,kwargs


def test_real_strict_raw_replay_retains_cropped_edges_on_fixed_source_grid(raw_road):
    _,kwargs=raw_road
    result=measure_road_frame(**kwargs)
    assert result.audit['retained_by_side']['left']>=4
    assert result.audit['retained_by_side']['right']>=4
    assert all(abs(k['strict_measurement']['offset_m'])<=.05 for k in result.knots)
    assert all(k['source_route_s']==k['source_index']*.5 for k in result.knots)
    assert assemble([result]).road is not None
    shifted=measure_road_frame(**dict(kwargs,current_s=.13))
    assert shifted.knots==result.knots


def test_raw_hash_alignment_and_packed_corruption_fail_closed(raw_road,tmp_path):
    data,kwargs=raw_road
    with pytest.raises(ValueError,match='hash mismatch'):
        measure_road_frame(**dict(kwargs,payload_sha256='0'*64))
    with pytest.raises(ValueError,match='semantic sensor stamp'):
        measure_road_frame(**dict(kwargs,semantic_sensor_stamp=FrameStamp(9,.95,'narrow')))
    broken={k:v.copy() for k,v in data.items()};broken['packed'][0,0,0]^=1
    path=tmp_path/'broken.npz';np.savez_compressed(path,**broken)
    with pytest.raises(ValueError,match='packed model frame'):
        measure_road_frame(**dict(kwargs,payload=path,payload_sha256=sha256_file(path)))


def test_edge_outside_native_model_crop_is_not_observed(raw_road,tmp_path):
    data,kwargs=raw_road
    changed={k:v.copy() for k,v in data.items()}
    changed['model_to_source']=np.array([[.1,0.,1000.],[0.,1.,480.],[0.,0.,1.]])
    changed['packed']=pack_model_rgb(changed['rgb'],changed['model_to_source'])
    path=tmp_path/'no-edge-crop.npz';np.savez_compressed(path,**changed)
    result=measure_road_frame(**dict(kwargs,payload=path,payload_sha256=sha256_file(path)))
    assert sum(result.audit['strict_measured_by_side'].values())>0
    assert not result.knots
    assert any(key.startswith('model_crop:') for key in result.audit['rejections'])
