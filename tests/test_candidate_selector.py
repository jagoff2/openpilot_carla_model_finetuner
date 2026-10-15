import ast
import hashlib
import json
from types import SimpleNamespace

import pytest

from lane_release import local_candidate as candidate
from lane_release.selector_patch import patch_selector


class Params:
    def __init__(self):
        self.data = {'ModelManager_ActiveBundle': {'ref': 'previous'}}
    def get(self, key):
        return self.data.get(key)
    def put(self, key, value):
        self.data[key] = value
    def remove(self, key):
        self.data.pop(key, None)


def installed(tmp_path, monkeypatch):
    models, artifacts = [], {}
    for role in ('vision', 'policy'):
        model = {'type': role}
        for kind in ('artifact', 'metadata'):
            name = f'{role}-{kind}.pkl'
            (tmp_path / name).write_bytes(name.encode())
            digest = hashlib.sha256(name.encode()).hexdigest()
            artifacts[name] = digest
            model[kind] = {'fileName': name, 'downloadUri': {'sha256': digest}}
        models.append(model)
    bundle = dict(ref=candidate.CANDIDATE_REF, displayName=candidate.DISPLAY_NAME, runner='tinygrad', models=models)
    (tmp_path / candidate.DESCRIPTOR_FILE).write_text(json.dumps(bundle))
    (tmp_path / candidate.RECEIPT_FILE).write_text(json.dumps(dict(passed=True, runtime_ready=True, artifacts=artifacts)))
    monkeypatch.setattr(candidate, '_root', lambda: tmp_path)
    return bundle


def test_explicit_selection_rehashes_and_keeps_prior_model_on_failure(tmp_path, monkeypatch):
    bundle = installed(tmp_path, monkeypatch)
    params = Params()
    assert candidate.validated_bundle() == bundle
    assert params.get('ModelManager_ActiveBundle')['ref'] == 'previous'
    with pytest.raises(RuntimeError, match='offroad'):
        candidate.select_candidate(params, offroad=False)
    (tmp_path / 'vision-artifact.pkl').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='integrity'):
        candidate.select_candidate(params, offroad=True)
    assert params.get('ModelManager_ActiveBundle')['ref'] == 'previous'
    installed(tmp_path, monkeypatch)
    params.put('ModelManager_DownloadIndex', 56)
    candidate.select_candidate(params, offroad=True)
    assert candidate.is_selected(params)
    assert params.get('ModelManager_DownloadIndex') is None


def test_actual_picker_patch_has_no_new_toggle_and_selects_locally():
    from pathlib import Path
    source_path = Path(__file__).resolve().parents[1] / 'releases/cd210/20261015T095217636853Z-step24-candidate-eb2b5c3d/selector-install/iq_panels.original.py'
    if not source_path.is_file():
        pytest.skip('device UI snapshot not available')
    original = source_path.read_text()
    source = patch_selector(original)
    assert source.count('toggle_item') == original.count('toggle_item')
    assert patch_selector(source) == source
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelsLayout')
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_get_folders', '_on_model_selected')]
    selected, resets, values = [], [], []
    state = SimpleNamespace(params=Params(), is_offroad=lambda: True,
                            sm={'iqModelManager': SimpleNamespace(availableBundles=[])})
    entries = [dict(ref='cd210-centerline-s71-' + 'a' * 12, display_name='CD210 Centerline s71', short_name='step 71'),
               dict(ref=candidate.CANDIDATE_REF, display_name=candidate.DISPLAY_NAME, short_name='step 24')]
    def select(params, *, ref, offroad):
        selected.append(ref)
        return dict(displayName=next(item['display_name'] for item in entries if item['ref'] == ref))
    env = dict(centerline_candidates=lambda: entries, is_centerline_candidate_ref=candidate.is_candidate_ref,
        CANDIDATE_REF='full-frame-fov-s110-v1', FULL_FRAME_CANDIDATE_NAME='Full Frame FOV',
        ui_state=state, DialogResult=SimpleNamespace(CONFIRM=1), tr=lambda s:s, _is_c3x=lambda:True,
        select_centerline_candidate=select, get_cached_model_bundles=lambda p:[],
        _full_frame_candidate_status_label=lambda:'ready',
        PickerItem=lambda ref, data:SimpleNamespace(ref=ref, data=data),
        PickerGroup=lambda name, items:SimpleNamespace(name=name, items=items))
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<picker>','exec'),env)
    self = SimpleNamespace(model_dialog=SimpleNamespace(selection_ref=candidate.CANDIDATE_REF),
        full_frame_candidate_item=SimpleNamespace(action_item=SimpleNamespace(set_state=lambda x:None)),
        current_model_item=SimpleNamespace(action_item=SimpleNamespace(set_value=values.append)),
        _show_reset_params_dialog=lambda:resets.append(True))
    groups = env['_get_folders'](self)
    experimental = next(g for g in groups if g.name == 'Experimental')
    assert [i.ref for i in experimental.items] == [entries[0]['ref'], candidate.CANDIDATE_REF, 'full-frame-fov-s110-v1']
    env['_on_model_selected'](self,1)
    assert selected == [candidate.CANDIDATE_REF] and values == [candidate.DISPLAY_NAME]
    self.model_dialog = SimpleNamespace(selection_ref=entries[0]['ref'])
    env['_on_model_selected'](self, 1)
    assert selected == [candidate.CANDIDATE_REF, entries[0]['ref']]
    assert values == [candidate.DISPLAY_NAME, 'CD210 Centerline s71']
    state.is_offroad=lambda:False
    self.model_dialog=SimpleNamespace(selection_ref=candidate.CANDIDATE_REF)
    env['_on_model_selected'](self,1)
    assert selected == [candidate.CANDIDATE_REF, entries[0]['ref']]


def test_existing_step24_device_picker_upgrades_idempotently():
    from pathlib import Path
    snapshot = Path(__file__).resolve().parents[1] / 'reports/c3x-iq-panels-step71-before.py'
    if not snapshot.is_file():
        pytest.skip('current device UI snapshot not available')
    original = snapshot.read_text(encoding='utf-8')
    source = patch_selector(original)
    assert source.count('toggle_item') == original.count('toggle_item')
    assert 'CENTERLINE_CANDIDATE_REF' not in source
    assert patch_selector(source) == source
    ast.parse(source)
    with pytest.raises(ValueError, match='incomplete'):
        patch_selector(source.replace("        }) for item in centerline_candidates()],", "        }) for item in []],"))
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelsLayout')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_handle_current_model_clicked')
    assignment = next(node for node in method.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'active_ref' for target in node.targets))
    expression = compile(ast.Expression(body=assignment.value), '<active-ref>', 'eval')
    params = Params()
    env = dict(selected_centerline_ref=candidate.selected_ref, ui_state=SimpleNamespace(params=params),
               is_candidate_selected=lambda _:False, CANDIDATE_REF='full-frame',
               self=SimpleNamespace(_has_active_bundle_param=lambda:True,
                                    model_manager=SimpleNamespace(activeBundle=SimpleNamespace(ref='stock'))))
    assert eval(expression, env) == 'stock'
    for ref in (candidate.CANDIDATE_REF, 'cd210-centerline-s71-' + 'a' * 12):
        params.put('ModelManager_ActiveBundle', dict(ref=ref))
        assert eval(expression, env) == ref


def test_multi_candidate_identity_and_selection(tmp_path, monkeypatch):
    old_bundle = installed(tmp_path, monkeypatch)
    ref = 'cd210-centerline-s71-' + 'a' * 12
    bundle = dict(old_bundle, ref=ref, displayName='CD210 Centerline s71')
    payload = dict(step=71, selected_checkpoint_sha256='a' * 64, bundle=dict(ref=ref))
    payload_path = tmp_path / (ref + '.payload.json')
    payload_path.write_text(json.dumps(payload))
    report = json.loads((tmp_path / candidate.RECEIPT_FILE).read_text())
    report.update(frame_count=120, source_payload_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest())
    (tmp_path / (ref + '.bundle.json')).write_text(json.dumps(bundle))
    (tmp_path / (ref + '.device-report.json')).write_text(json.dumps(report))
    assert [entry['ref'] for entry in candidate.candidates()] == [ref, candidate.CANDIDATE_REF]
    assert candidate.validated_bundle() == old_bundle
    params = Params()
    candidate.select_candidate(params, ref=ref, offroad=True)
    assert candidate.selected_ref(params) == ref
    assert candidate.is_selected(params, ref)
    candidate.select_candidate(params, offroad=True)
    assert candidate.selected_ref(params) == candidate.CANDIDATE_REF
    payload_path.write_text(json.dumps(dict(payload, step=72)))
    with pytest.raises(RuntimeError, match='identity'):
        candidate.select_candidate(params, ref=ref, offroad=True)
    assert candidate.selected_ref(params) == candidate.CANDIDATE_REF
    assert [entry['ref'] for entry in candidate.candidates()] == [candidate.CANDIDATE_REF]


@pytest.mark.parametrize('ref', ['../bad', 'cd210-centerline-s71-' + 'a' * 12 + '/bad', None])
def test_candidate_reference_cannot_escape_model_root(tmp_path, monkeypatch, ref):
    monkeypatch.setattr(candidate, '_root', lambda: tmp_path)
    assert not candidate.is_candidate_ref(ref)
    with pytest.raises(RuntimeError, match='reference'):
        candidate.validated_bundle(ref)
