"""Photometric alternatives retain actual surface, crop and precision gates."""
import json
from pathlib import Path

import numpy as np
import pytest

from lane_behavior.curb_evidence import (_upper_lip_color_proof,inspect_curb_surface,
    surveyed_reference,_cropped_surface_points,_cropped_pixel_colors)
from lane_dataset.config import sha256_file


def color_fixture():
    # Separate synthetic negative-control fixture, never a saved camera proof.
    along=np.tile(np.linspace(-.7,.7,8),4)
    outward=np.repeat([.03,.17,.35,.65],8)
    n=len(along);lip=outward<.2
    rgb=np.where(lip[:,None],[190,190,190],[140,180,100]).astype(np.uint8)
    packed=np.where(lip[:,None],[179,128,128],[160,95,115]).astype(np.uint8)
    xy=np.c_[np.arange(n),np.arange(n)]
    return dict(xy=xy,tags=np.where(lip,25,10),lateral=outward,along=along,
        height=np.full(n,.152),sign=1.,toe_offset=0.,colors=(rgb,packed,xy.copy()))


def test_coherent_source_and_model_visible_level_lip_edge_passes():
    proof=_upper_lip_color_proof(**color_fixture())
    assert proof['passed']
    assert proof['minimum_rgb_and_packed_contrast']>=12.
    assert proof['maximum_lip_plane_residual_m']<.005
    assert len(proof['lip_source_pixels'])==len(proof['lip_model_pixels'])


@pytest.mark.parametrize('change', ['flat_rgb','packed_edge_absent','static_object','pole',
    'hidden_verge','one_half_only','inconsistent_edge','raised_object','nonplanar_lip'])
def test_lip_alternative_rejects_missing_ambiguous_or_noncurb_support(change):
    args=color_fixture();lip=args['lateral']<.2
    rgb,packed,model=args['colors']
    if change=='flat_rgb':rgb[:]=180
    elif change=='packed_edge_absent':packed[:]=128
    elif change=='static_object':args['tags'][~lip]=20
    elif change=='pole':args['tags'][~lip]=6
    elif change=='hidden_verge':args['lateral'][~lip]=1.5
    elif change=='one_half_only':args['tags'][(~lip)&(args['along']<0)]=6
    elif change=='inconsistent_edge':
        packed[(~lip)&(args['along']>0)]=[200,160,140]
    elif change=='raised_object':args['height'][~lip]=.8
    elif change=='nonplanar_lip':args['height'][np.flatnonzero(lip)[0]]+=.015
    proof=_upper_lip_color_proof(**args)
    assert not proof['passed'], (change,proof)


@pytest.fixture(scope='module')
def late_native_frames():
    root=Path(r'G:\lane-centering-dataset-v1\reports\unmarked\captures\20261015T131042-60299648')
    if not root.exists():pytest.skip('saved moving native road51 qualification capture required')
    report=json.loads((root/'capture-report.json').read_text());bindings=report['retained_scene_bindings']
    geometry=surveyed_reference(root/bindings['survey_replay']['path'],bindings['survey_replay']['sha256'],
        raw_survey_path=root/bindings['raw_survey']['path'])
    frames={r['index']:r for r in map(json.loads,(root/'evidence-frames.jsonl').read_text().splitlines())}
    result={}
    for index in (450,550,650):
        meta=frames[index]['cameras']['wide'];path=root/meta['payload']
        assert sha256_file(path)==meta['payload_sha256']
        with np.load(path,allow_pickle=False) as saved:data={k:saved[k] for k in saved.files}
        points=_cropped_surface_points(data)
        result[index]=(data,points,_cropped_pixel_colors(data,points[0]))
    return geometry,result


@pytest.mark.parametrize('index,station', [(450,132.5),(550,139.5),(650,155.5)])
def test_bound_late_native_payload_recovers_actual_lip_without_relaxing_geometry(late_native_frames,index,station):
    (reference,audit),frames=late_native_frames;data,points,colors=frames[index]
    proof=inspect_curb_surface(data,reference,station,'left',model_crop_only=True,
        _cropped_points=points,_cropped_colors=colors)
    assert proof['surface_hypothesis'] and proof['upper_lip_color_proof']['passed']
    assert proof['face_to_pavement_luma_contrast']<12.
    assert proof['rgb_contrast']>=12.
    assert proof['maximum_surface_fit_residual_m']<=.005
    budget=abs(proof['toe_offset_from_reference_m'])+audit['left']['maximum_withheld_error_m']+audit['maximum_toe_uncertainty_m']+proof['maximum_surface_fit_residual_m']
    assert budget<=.05
    assert not proof['independent_collision_agreement_verified'] and not proof['training_ready']
    assert proof['upper_lip_color_proof']['maximum_lip_plane_residual_m']<=.005
    assert all(h['source_rgb_contrast']>=12. and h['packed_yuv_contrast']>=12.
        for h in proof['upper_lip_color_proof']['halves'])


def test_lamp_occlusion_cannot_substitute_for_the_missing_native_verge(late_native_frames):
    (reference,_),frames=late_native_frames;data,points,colors=frames[450]
    proof=inspect_curb_surface(data,reference,129.,'left',model_crop_only=True,
        _cropped_points=points,_cropped_colors=colors)
    assert proof['surface_hypothesis']
    assert proof['rgb_contrast']<12.
    assert not proof['upper_lip_color_proof']['passed']


def test_actual_geometry_with_uniform_color_cannot_pass_by_depth_alone(late_native_frames):
    (reference,_),frames=late_native_frames;data,points,colors=frames[650]
    changed=dict(data);changed['rgb']=np.full_like(data['rgb'],180)
    changed_points=(*points[:3],np.full_like(points[3],171))
    blank=(np.full_like(colors[0],180),np.full_like(colors[1],128),colors[2])
    proof=inspect_curb_surface(changed,reference,155.5,'left',model_crop_only=True,
        _cropped_points=changed_points,_cropped_colors=blank)
    assert proof['surface_hypothesis'] and proof['rgb_contrast']<12.
    assert not proof['upper_lip_color_proof']['passed']
