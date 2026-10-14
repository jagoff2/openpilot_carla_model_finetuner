import ast
import json
import shutil

import numpy as np
import pytest

from lane_dataset import settings
from lane_dataset.config import Config,PROJECT
from lane_dataset.preprocessing import DEPENDENCY_ROOT,MODEL_SIZE,RGB_PRODUCER_SOURCE
from lane_training.cd210 import DEFAULT_CHECKPOINT,PINNED_FILES,verify_checkpoint
from lane_training.fetch_cd210 import EXPORT_DIR,EXPORTS,fetch_bundle


def test_settings_precedence_and_relative_paths(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,'PROJECT',tmp_path)
    (tmp_path/'config.local.json').write_text(json.dumps(dict(dataset_root='configured/corpus')))
    monkeypatch.delenv('LANE_DATASET_ROOT',raising=False)
    assert settings.path_setting('dataset_root','LANE_DATASET_ROOT','fallback')==tmp_path/'configured/corpus'
    monkeypatch.setenv('LANE_DATASET_ROOT','environment/corpus')
    assert settings.path_setting('dataset_root','LANE_DATASET_ROOT','fallback')==tmp_path/'environment/corpus'


def test_no_sibling_project_is_required_for_preprocessing_or_default_model():
    assert DEPENDENCY_ROOT==PROJECT/'lane_reference'
    assert DEFAULT_CHECKPOINT==PROJECT/'models/cd210/bundle/bundle.json'
    assert Config().model_sha256==EXPORTS['policy']['sha256']
    assert Config().model_path.endswith('driving_policy.onnx')


def test_review_video_conversion_helpers_are_bundled_and_work():
    names={'unpack_six_plane_yuv420','packed_to_rgb_bt601_limited'}
    tree=ast.parse(RGB_PRODUCER_SOURCE.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    namespace=dict(np=np,MODEL_SIZE=MODEL_SIZE)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(RGB_PRODUCER_SOURCE),'exec'),namespace)
    packed=np.empty((6,128,256),np.uint8);packed[:4]=16;packed[4:]=128
    rgb=namespace['packed_to_rgb_bt601_limited'](packed)
    assert rgb.shape==(256,512,3) and not rgb.any()


def test_exact_metadata_generated_without_private_server_or_compiled_pkls(tmp_path):
    for entry in EXPORTS.values():shutil.copyfile(EXPORT_DIR/entry['file'],tmp_path/entry['file'])
    fetch_bundle(tmp_path/'bundle')
    report=verify_checkpoint(tmp_path/'bundle/bundle.json')
    assert report['passed'] and not report['compiled_artifacts_verified']
    assert len(list((tmp_path/'bundle').glob('*_metadata.pkl')))==2
    assert not list((tmp_path/'bundle').glob('*_tinygrad.pkl'))
    for name,entry in report['files'].items():
        if name.endswith('_metadata.pkl'):assert entry['verified'] and entry['sha256']==PINNED_FILES[name]


def test_regenerated_metadata_still_rejects_modified_bytes(tmp_path):
    for entry in EXPORTS.values():shutil.copyfile(EXPORT_DIR/entry['file'],tmp_path/entry['file'])
    fetch_bundle(tmp_path/'bundle')
    path=tmp_path/'bundle/driving_policy_c210m_metadata.pkl';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):verify_checkpoint(tmp_path/'bundle/bundle.json')


def test_seed_configuration_is_not_a_carla_asset_or_recorded_dataset():
    seed=json.loads((PROJECT/'pipeline/assets/town06-seed.json').read_text())
    assert seed['town']=='Town06' and seed['start']==dict(road_id=40,lane_id=-4,s=405.)
    assert 'route' not in seed and 'labels' not in seed and not seed['rendered_qualification']
