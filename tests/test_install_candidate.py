import hashlib
import json

import pytest

from lane_release.install_candidate import candidate_index, cached_bundle, immutable_copy, verified_package


def make_package(tmp_path):
    compiled = tmp_path / 'compiled'
    compiled.mkdir()
    ref = 'cd210-centerline-s71-' + 'a' * 12
    roles, models, artifacts = {}, [], {}
    for role in ('vision', 'policy'):
        entry, model = {}, dict(type=role)
        for kind in ('artifact', 'metadata'):
            name = f's71-{role}-{kind}.pkl'
            content = name.encode()
            (compiled / name).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            artifacts[name] = digest
            entry[kind] = name
            model[kind] = dict(fileName=name, downloadUri=dict(sha256=digest))
        roles[role] = entry
        models.append(model)
    bundle = dict(ref=ref, runner='tinygrad', models=models, overrides=[dict(key='folder', value='original')])
    manifest = dict(step=71, selected_checkpoint_sha256='a' * 64, bundle=bundle, roles=roles)
    (tmp_path / 'payload.json').write_text(json.dumps(manifest))
    report = dict(passed=True, runtime_ready=True, frame_count=120, artifacts=artifacts,
                  source_payload_sha256=hashlib.sha256((tmp_path / 'payload.json').read_bytes()).hexdigest())
    (compiled / 'device-report.json').write_text(json.dumps(report))
    (compiled / 'bundle.json').write_text(json.dumps(bundle))
    return manifest, report, bundle


def test_explicit_checkpoint_identity_and_device_results_bound(tmp_path):
    expected = make_package(tmp_path)
    assert verified_package(tmp_path, 71, 'a' * 64) == expected
    with pytest.raises(RuntimeError, match='identity'):
        verified_package(tmp_path, 72, 'a' * 64)
    with pytest.raises(RuntimeError, match='identity'):
        verified_package(tmp_path, 71, 'b' * 64)
    report_path = tmp_path / 'compiled/device-report.json'
    report = expected[1]
    report['runtime_ready'] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='verification'):
        verified_package(tmp_path, 71, 'a' * 64)


@pytest.mark.parametrize('mutation', ['artifact', 'descriptor_hash', 'payload', 'extra_file', 'role_filename'])
def test_package_corruption_rejected(tmp_path, mutation):
    manifest, report, bundle = make_package(tmp_path)
    if mutation == 'artifact':
        (tmp_path / 'compiled/s71-vision-artifact.pkl').write_bytes(b'changed')
    elif mutation == 'descriptor_hash':
        bundle['models'][0]['artifact']['downloadUri']['sha256'] = '0' * 64
        (tmp_path / 'compiled/bundle.json').write_text(json.dumps(bundle))
    elif mutation == 'payload':
        manifest['step'] = 24
        (tmp_path / 'payload.json').write_text(json.dumps(manifest))
    elif mutation == 'extra_file':
        report['artifacts']['../other'] = '0' * 64
        (tmp_path / 'compiled/device-report.json').write_text(json.dumps(report))
    else:
        bundle['models'][0]['artifact']['fileName'] = 'other.pkl'
        (tmp_path / 'compiled/bundle.json').write_text(json.dumps(bundle))
    with pytest.raises(RuntimeError):
        verified_package(tmp_path, 71, 'a' * 64)


def test_index_collision_keeps_existing_candidates_and_is_stable(tmp_path):
    old = dict(ref='old-step24', index=65024)
    (tmp_path / 'old.bundle.json').write_text(json.dumps(old))
    cache = dict(bundles=[old, dict(ref='another-model', index=65071)])
    assert candidate_index(tmp_path, cache, 'new-step71', 71) == 65000
    new = dict(ref='new-step71', index=65000)
    (tmp_path / 'new.bundle.json').write_text(json.dumps(new))
    assert candidate_index(tmp_path, cache, 'new-step71', 71) == 65000
    assert json.loads((tmp_path / 'old.bundle.json').read_text()) == old
    with pytest.raises(RuntimeError, match='collision'):
        candidate_index(tmp_path, dict(bundles=[dict(ref='conflicting', index=65024)]), 'new', 72)


def test_immutable_copy_is_idempotent_and_never_replaces_prior_bytes(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.write_bytes(b'original candidate')
    immutable_copy(source, target)
    immutable_copy(source, target)
    source.write_bytes(b'new candidate')
    with pytest.raises(RuntimeError, match='refusing'):
        immutable_copy(source, target)
    assert target.read_bytes() == b'original candidate'
    assert not list(tmp_path.glob('*.tmp'))


def test_cached_bundle_is_distinct_experimental_and_does_not_mutate_payload(tmp_path):
    manifest, _, bundle = make_package(tmp_path)
    original = json.loads(json.dumps(bundle))
    installed = cached_bundle(bundle, 65071, 71, 'a' * 64)
    assert bundle == original == manifest['bundle']
    assert installed['index'] == 65071
    assert installed['displayName'] == 'CD210 Centerline s71 (aaaaaaaa)'
    assert installed['overrides'] == [dict(key='folder', value='Experimental')]
    assert all(m[k]['downloadProgress']['status'] == 'cached'
               for m in installed['models'] for k in ('artifact', 'metadata'))
