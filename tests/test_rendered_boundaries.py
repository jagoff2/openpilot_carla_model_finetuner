import gzip
import json
import numpy as np
import pytest

from lane_dataset.rendered_boundaries import stripe_center, summarize_frames, check_spatial_resolution
from lane_dataset.admission import rendered_boundary_status
from lane_dataset.config import sha256_file, write_json


@pytest.mark.parametrize('shift,passed',[(0.,True),(.12,False)])
def test_resolved_paint_beyond_unmarked_junction_preserves_agreement_gate(shift,passed):
    from lane_dataset.geometry import LaneReference
    from lane_dataset.rendered_boundaries import measure_camera
    s=np.linspace(0.,90.,181);center=np.c_[s,np.zeros(len(s)),np.zeros(len(s))]
    reference=LaneReference(center+[0.,-1.8,0.],center+[0.,1.8,0.],s)
    k=np.array([[6000.,0.,1000.],[0.,6000.,400.],[0.,0.,1.]])
    camera=np.array([[0.,1.,0.,0.],[0.,0.,-1.,1.5],[1.,0.,0.,0.],[0.,0.,0.,1.]])
    yy,xx=np.indices((800,2000));distance=np.full(yy.shape,1000.)
    np.divide(9000.,yy-400,out=distance,where=yy>400)
    lateral=(xx-1000.)*distance/6000.
    paint=((np.abs(lateral-(-1.8+shift))<=.075)|(np.abs(lateral-(1.8+shift))<=.075))&(distance>=35.)&(distance<=50.)
    tags=np.where(paint,24,1).astype(np.uint8)
    measured=measure_camera(reference=reference,current_s=0.,tags=tags,depth=distance,
        camera_from_world=camera,intrinsics=k,depth_mode='forward',road_line_tag=24)
    result=summarize_frames([dict(frame_id=1,cameras={'narrow':measured})])
    assert result['passed'] is passed
    assert all(n>=3 for n in result['measurements']['samples_per_frame'][0].values())
    if not passed:assert 'rendered_boundary_disagreement_above_five_cm' in result['errors']


def stripe(offset=.0):
    offsets=np.linspace(-.4,.4,161)
    mask=np.abs(offsets-offset)<=.05+1e-9
    return offsets,mask,np.zeros(161)


def test_rendered_paint_offset_is_measured_instead_of_clamped_to_map():
    offsets,mask,depth=stripe(.12)
    result=stripe_center(offsets,mask,depth)
    assert result["measured"]
    assert result["offset_m"]==pytest.approx(.12)
    assert stripe_center(offsets,mask,depth+.2)["reason"]=="paint_depth_disagreement"


def test_declared_width_rejects_fragments_without_hiding_displaced_full_stripes():
    offsets,mask,depth=stripe(.12)
    fragment=(offsets>=-.07)&(offsets<=-.035)
    result=stripe_center(offsets,fragment,depth,expected_width_m=.125)
    assert not result['measured']
    assert result['reason']=='paint_width_unresolved'
    for shift in (0.,.06,.12):
        offsets,mask,depth=stripe(shift)
        measured=stripe_center(offsets,mask,depth,expected_width_m=.125)
        assert measured['measured']
        assert measured['offset_m']==pytest.approx(shift)
    frame=frame_fixture(.06)
    for sample in frame['cameras']['wide']['samples']:
        sample['expected_marking_width_m']=.125
    assert not summarize_frames([frame])['passed']


def test_spatial_resolution_gate_is_independent_of_measured_agreement():
    for offset in (0.,.12):
        offsets,mask,depth=stripe(offset)
        result=stripe_center(offsets,mask,depth)
        assert not check_spatial_resolution(result,.04)['measured']
        assert not check_spatial_resolution(result,.02)['measured']
        assert check_spatial_resolution(result,.008)['measured']
        assert check_spatial_resolution(result,.008)['offset_m']==pytest.approx(offset)


def frame_fixture(offset=0.):
    offsets,mask,residual=stripe(offset)
    samples=[]
    for side in ("left","right"):
        for position in (5.,6.,7.):
            samples.append(dict(side=side,route_s=position,
                marking_bits=np.packbits(mask).tolist(),depth_residual_m=residual.tolist(),
                **stripe_center(offsets,mask,residual)))
    return dict(frame_id=10,index=0,cameras=dict(wide=dict(offsets_m=offsets.tolist(),samples=samples)))


def test_missing_side_or_displaced_paint_cannot_pass():
    assert summarize_frames([frame_fixture()])["passed"]
    displaced=summarize_frames([frame_fixture(.12)])
    assert not displaced["passed"]
    assert displaced["maximum_disagreement_m"]==pytest.approx(.12)
    frame=frame_fixture()
    frame["cameras"]["wide"]["samples"]=frame["cameras"]["wide"]["samples"][:3]
    assert not summarize_frames([frame])["passed"]
    assert not summarize_frames([])["passed"]


def test_precise_static_geometry_evidence_expires_and_cannot_come_from_future():
    complete=frame_fixture();missing=frame_fixture()
    missing['frame_id']=11;missing['cameras']['wide']['samples']=[]
    assert summarize_frames([complete,missing])['passed']
    missing['frame_id']=107
    assert not summarize_frames([complete,missing])['passed']
    missing['frame_id']=9
    assert not summarize_frames([missing,complete])['passed']
    forged=frame_fixture();forged['cameras']['wide']['samples'][0].update(proof_omitted='unresolved_no_evidence',measured=True)
    assert not summarize_frames([forged])['passed']


def test_admission_replays_bound_samples_and_rejects_missing_frames(tmp_path):
    frames=[frame_fixture()]
    samples=tmp_path/"rendered-boundary-samples.json.gz"
    with gzip.open(samples,"wt") as stream:
        json.dump(frames,stream)
    report=summarize_frames(frames)
    report["samples_sha256"]=sha256_file(samples)
    write_json(tmp_path/"rendered-boundary-report.json",report)
    manifest=dict(files={p.name:sha256_file(p) for p in tmp_path.iterdir()})
    assert rendered_boundary_status(tmp_path,manifest,dict(frame_id=np.array([10])))["passed"]
    with pytest.raises(ValueError,match="every retained frame"):
        rendered_boundary_status(tmp_path,manifest,dict(frame_id=np.array([10,11])))
    frames[0]["cameras"]["wide"]["samples"][0]["offset_m"] = 0.2
    assert not summarize_frames(frames)["passed"]
