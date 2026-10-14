from dataclasses import replace
import numpy as np
import pytest
from lane_dataset.visibility import CameraObservation,probe_lane_evidence
from lane_dataset.admission import _verify_pixel_claims
from lane_dataset.rendered_boundaries import edge_center,summarize_frames


def scene():
    gray=np.full((320,640),45,np.uint8);gray[:,101:]=110
    road=np.zeros_like(gray,bool);road[:,:101]=True
    return CameraObservation(0,0.,'narrow',np.repeat(gray[:,:,None],3,axis=2),
        np.full(gray.shape,10.,np.float32),np.eye(4),np.diag([100.,100.,1.]),
        np.eye(3,dtype=np.float32),depth_validated=True,luma=gray,
        marking_mask=np.zeros_like(road),road_mask=road,road_exterior_mask=~road)


def probe(obs):
    return probe_lane_evidence(obs,[[10.,10.,10.]],['R0'],[10.])


def packed(obs):
    image=obs.luma[:256,:512]
    value=np.zeros((6,128,256),np.uint8)
    value[0],value[1],value[2],value[3]=image[::2,::2],image[1::2,::2],image[::2,1::2],image[1::2,1::2]
    return value


def test_curb_requires_rgb_depth_semantic_transition_and_replays():
    obs=scene();evidence=probe(obs)
    assert evidence.crop_visible.tolist()==[True]
    assert evidence.crop_boundary_kind.tolist()==[2]
    _verify_pixel_claims(evidence,packed(obs),np.zeros(3),np.array([[10.,10.,10.]]))
    broken=packed(obs);broken[:4]=45
    with pytest.raises(ValueError,match='edge contrast'):
        _verify_pixel_claims(evidence,broken,np.zeros(3),np.array([[10.,10.,10.]]))
    for changed in (replace(obs,luma=np.full_like(obs.luma,45)),
                    replace(obs,road_exterior_mask=np.zeros_like(obs.road_mask)),
                    replace(obs,depth_m=np.full_like(obs.depth_m,3.)),
                    replace(obs,road_mask=np.zeros_like(obs.road_mask))):
        assert not probe(changed).crop_visible.any()


def test_edge_exterior_depth_and_crop_bounds_cannot_be_bypassed():
    obs=scene();depth=obs.depth_m.copy();depth[:,101:]=3.
    assert not probe(replace(obs,depth_m=depth)).crop_visible.any()
    warp=np.eye(3,dtype=np.float32);warp[0,2]=300.
    evidence=probe(replace(obs,model_to_source=warp))
    assert evidence.source_visible[0] and not evidence.crop_visible[0]
    good=probe(obs)
    with pytest.raises(ValueError,match='exterior depth'):
        _verify_pixel_claims(replace(good,crop_edge_other_depth_residual_m=np.array([7.])),
                             packed(obs),np.zeros(3),np.array([[10.,10.,10.]]))


def test_metric_edge_measurement_retains_offset_and_five_cm_limit():
    offsets=np.linspace(-.4,.4,161);road=offsets<.12;other=~road
    gray=np.where(road,45,110);depth=np.zeros(161)
    result=edge_center(offsets,road,other,gray,depth,'right')
    assert result['measured'] and result['offset_m']==pytest.approx(.1175)
    sample=dict(side='right',route_s=10.,marking_bits=np.packbits(np.zeros(161,bool)).tolist(),
        road_bits=np.packbits(road).tolist(),exterior_bits=np.packbits(other).tolist(),
        luma=gray.tolist(),depth_residual_m=depth.tolist(),**result)
    report=summarize_frames([dict(frame_id=0,cameras=dict(wide=dict(offsets_m=offsets.tolist(),samples=[sample])) )])
    assert 'rendered_boundary_disagreement_above_five_cm' in report['errors']
    assert not edge_center(offsets,road,other,np.full(161,45),depth,'right')['measured']
    assert not edge_center(offsets,road,other,gray,depth+.2,'right')['measured']
    assert not edge_center(offsets,road,other,gray,depth,'left')['measured']
