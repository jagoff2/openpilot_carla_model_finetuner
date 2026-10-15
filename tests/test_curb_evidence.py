import json
from pathlib import Path

import numpy as np
import pytest

from lane_behavior.curb_evidence import inspect_curb_surface,surveyed_reference,measure_curb_road_frame
from lane_behavior.road_evidence import assemble_observed_road
from lane_behavior.evidence import FrameStamp
from lane_dataset.config import sha256_file
from lane_dataset.geometry import LaneReference


@pytest.fixture(scope='module')
def native_payload():
    root=Path(r'G:\lane-centering-dataset-v1\reports\unmarked\qualification\20261015T120245-053fb75b')
    plan=Path(r'G:\lane-centering-dataset-v1\reports\unmarked\unmarked-native-82e72aca73e5be2b3c08-Town04-road51-lane-1-inner-curb-obstruction-v2-candidate-plan.json')
    if not root.exists() or not plan.exists():pytest.skip('immutable native diagnostic fixture is not installed')
    report=json.loads((root/'probe-report.json').read_text())
    meta=report['frame_records'][0]['cameras']['wide']
    assert sha256_file(root/meta['payload'])==meta['payload_sha256']
    with np.load(root/meta['payload'],allow_pickle=False) as saved:data={key:saved[key] for key in saved.files}
    route=json.loads(plan.read_text())['tasks'][0]['route'];road=route['unmarked_road']
    reference=LaneReference(road['road_left_edge_xyz'],road['road_right_edge_xyz'],route['arc_m'])
    return data,reference


def test_bound_native_raised_face_is_visible_beyond_flat_semantic_seam(native_payload):
    data,ref=native_payload
    row=inspect_curb_surface(data,ref,88.,'left')
    assert row['surface_hypothesis']
    assert row['toe_offset_from_reference_m']==pytest.approx(-.4434,abs=.005)
    assert row['rgb_contrast']>=12
    assert row['p95_surface_fit_residual_m']<.002
    assert not row['training_ready'] and not row['independent_collision_agreement_verified']


def test_native_crop_availability_is_separate_from_full_source_visibility(native_payload):
    data,ref=native_payload
    assert not inspect_curb_surface(data,ref,88.,'left',model_crop_only=True)['surface_hypothesis']
    source=inspect_curb_surface(data,ref,100.,'left')
    crop=inspect_curb_surface(data,ref,100.,'left',model_crop_only=True)
    assert crop['surface_hypothesis'] and crop['rgb_contrast']>=12
    assert abs(crop['toe_offset_from_reference_m']-source['toe_offset_from_reference_m'])<.002
    # The crop uses a larger bounded metric patch to collect sparse native
    # samples; sampling density remains lower than the full source image.
    assert crop['face_samples']/crop['maximum_face_longitudinal_search_extent_m'] < source['face_samples']/source['maximum_face_longitudinal_search_extent_m']


def test_same_rgb_shadow_texture_on_flat_depth_cannot_create_raised_curb(native_payload):
    data,ref=native_payload
    # Explicit negative control: preserve actual RGB/semantic material seam,
    # replace geometry with a plane. This is a test perturbation, not saved data.
    changed=dict(data);h,w=data['depth_m'].shape
    y,x=np.indices((h,w));rays=np.stack((x+.5,y+.5,np.ones((h,w))),axis=-1)@np.linalg.inv(data['intrinsics']).T
    inverse=np.linalg.inv(data['camera_from_world']);directions=rays@inverse[:3,:3].T
    plane_z=float(ref._left(100.)[2]);origin=inverse[:3,3]
    changed['depth_m']=np.divide(plane_z-origin[2],directions[...,2],out=np.full((h,w),1000.),where=abs(directions[...,2])>1e-8).astype(np.float32)
    assert not inspect_curb_surface(changed,ref,100.,'left',model_crop_only=True)['surface_hypothesis']


def test_native_cropped_curb_knots_form_observed_road_without_hidden_origin(native_payload):
    root=Path(r'G:\lane-centering-dataset-v1\reports\unmarked')
    source=root/'qualification/20261015T120245-053fb75b'
    survey=root/'surveys/20261015T123950-057a8ba5-beveled-toe-replay.json'
    if not survey.exists():pytest.skip('independent native collision survey not installed')
    geometry=surveyed_reference(survey,sha256_file(survey))
    report=json.loads((source/'probe-report.json').read_text());record=report['frame_records'][0]
    frames=[]
    for camera,meta in record['cameras'].items():
        frames.append(measure_curb_road_frame(payload=source/meta['payload'],payload_sha256=meta['payload_sha256'],
            stamp=FrameStamp(record['frame_id'],record['timestamp_s'],camera),modality_stamps=meta['modality_stamps'],
            semantic_sensor_stamp=meta['separate_semantic_sensor_stamp'],depth_fixture=source/'depth-fixture.json',
            depth_fixture_sha256=report['depth_fixture_sha256'],survey=survey,survey_sha256=sha256_file(survey),
            current_s=geometry[0].project(record['ego']['position_world_m'])[0],surveyed_geometry=geometry))
    result=assemble_observed_road(frames,decision=FrameStamp(record['frame_id'],record['timestamp_s'],'narrow'),
        position_world=record['ego']['position_world_m'])
    assert result.road is not None and not result.reasons
    assert result.audit['distinct_observed_knots']['left']>=4
    # More actual face pixels can support a longer future span. Neither a
    # larger fitting patch nor a lip edge may fabricate the hidden ego origin.
    ego=np.asarray(record['ego']['position_world_m'])
    assert np.linalg.norm(result.road.reference.center_xyz[:,:2]-ego[:2],axis=1).min()>15.
    for f in frames:
        for k in f.knots:
            assert k['curb_surface_proof']['rgb_contrast']>=12
            assert k['curb_surface_proof']['conservative_toe_error_budget_m']<=.05


def test_modified_derived_survey_cannot_replace_independent_raw_collision_toes(native_payload,tmp_path):
    path=Path(r'G:\lane-centering-dataset-v1\reports\unmarked\surveys\20261015T123950-057a8ba5-beveled-toe-replay.json')
    if not path.exists():pytest.skip('independent native collision survey not installed')
    report=json.loads(path.read_text())
    row=next(r for r in report['rows'] if r['passed']);row['toe_xyz'][0]+=.1
    altered=tmp_path/'survey.json';altered.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='coordinates changed'):surveyed_reference(altered,sha256_file(altered))
