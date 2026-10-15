"""Install an explicitly pinned, verified CD210 candidate without activating it."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_copy(source, target):
    target = Path(target)
    if target.exists():
        if sha(source) != sha(target):
            raise RuntimeError(f'refusing to replace an existing candidate: {target}')
        return
    temporary = target.with_name(target.name + '.' + uuid4().hex + '.tmp')
    try:
        shutil.copyfile(source, temporary)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if sha(source) != sha(target):
                raise RuntimeError(f'refusing to replace an existing candidate: {target}')
    finally:
        temporary.unlink(missing_ok=True)


def _filename(name):
    if not isinstance(name, str) or not name or Path(name).name != name or '/' in name or '\\' in name:
        raise RuntimeError('invalid compiled artifact filename')
    return name


def verified_package(package, expected_step, expected_checkpoint_sha256):
    """Bind checkpoint, compiler verification, bundle, and every installed artifact."""
    if (type(expected_step) is not int or expected_step < 0
            or not re.fullmatch(r'[0-9a-f]{64}', expected_checkpoint_sha256)):
        raise RuntimeError('explicit checkpoint step and SHA256 are required')
    package = Path(package)
    compiled = package / 'compiled'
    report = json.loads((compiled / 'device-report.json').read_text())
    manifest = json.loads((package / 'payload.json').read_text())
    if not (report['passed'] is True and report['runtime_ready'] is True and report['frame_count'] >= 120
            and report['source_payload_sha256'] == sha(package / 'payload.json')
            and manifest['step'] == expected_step
            and manifest['selected_checkpoint_sha256'] == expected_checkpoint_sha256):
        raise RuntimeError('candidate verification or checkpoint identity failed')
    ref = f'cd210-centerline-s{expected_step}-{expected_checkpoint_sha256[:12]}'
    bundle = json.loads((compiled / 'bundle.json').read_text())
    if not (bundle['ref'] == ref and manifest['bundle']['ref'] == ref
            and bundle['runner'] == 'tinygrad' and len(bundle['models']) == 2
            and {m['type'] for m in bundle['models']} == {'vision', 'policy'}):
        raise RuntimeError('unexpected model reference or runner')
    declared = {}
    for model in bundle['models']:
        role = manifest['roles'][model['type']]
        for kind in ('artifact', 'metadata'):
            record = model[kind]
            name = _filename(record['fileName'])
            if name != role[kind] or name in declared:
                raise RuntimeError('compiled bundle differs from payload roles')
            declared[name] = record['downloadUri']['sha256']
    if declared != report['artifacts']:
        raise RuntimeError('compiled bundle and verification report differ')
    for name, digest in declared.items():
        if not re.fullmatch(r'[0-9a-f]{64}', digest) or sha(compiled / name) != digest:
            raise RuntimeError('compiled artifact integrity failed')
    return manifest, report, bundle


def candidate_index(root, cache, ref, step):
    """Keep old indices stable and allocate a free local index under the install lock."""
    entries = list((cache or {}).get('bundles', []))
    entries.extend(json.loads(path.read_text()) for path in Path(root).glob('*.bundle.json'))
    occupied = {}
    existing = set()
    for item in entries:
        index = item.get('index')
        if index is None:
            continue
        index = int(index)
        other = item.get('ref')
        if index in occupied and occupied[index] != other:
            raise RuntimeError(f'existing model index collision: {index}')
        occupied[index] = other
        if other == ref:
            existing.add(index)
    if len(existing) > 1:
        raise RuntimeError('existing candidate has inconsistent indices')
    if existing:
        return existing.pop()
    preferred = 65000 + step
    if preferred not in occupied:
        return preferred
    for index in range(65000, 2 ** 31):
        if index not in occupied:
            return index
    raise RuntimeError('no free model index')


def cached_bundle(bundle, index, step, checkpoint_sha256):
    bundle = deepcopy(bundle)
    bundle.update(index=index, internalName=f'CD210 Centerline s{step}', status='cached',
                  displayName=f'CD210 Centerline s{step} ({checkpoint_sha256[:8]})')
    overrides = bundle.setdefault('overrides', [])
    folders = [entry for entry in overrides if entry['key'] == 'folder']
    if len(folders) > 1:
        raise RuntimeError('duplicate candidate folder overrides')
    if folders:
        folders[0]['value'] = 'Experimental'
    else:
        overrides.append(dict(key='folder', value='Experimental'))
    for model in bundle['models']:
        for kind in ('artifact', 'metadata'):
            model[kind]['downloadProgress'] = dict(status='cached', progress=100)
    return bundle


@contextmanager
def installation_lock(root):
    import fcntl  # Device-side installer runs on Linux.
    with (Path(root) / '.cd210-candidate-install.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--expected-ui-sha256', required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    compiled = package / 'compiled'
    iqroot = Path('/data/iqpilot')
    root = Path('/data/media/0/models')
    sys.path[:0] = [str(iqroot / 'artifacts/package_runtime'), str(iqroot)]
    from iqpilot.common.params import Params
    from iqpilot.system.hardware import HARDWARE
    from iqpilot.selfdrive.iqmodeld.models.helpers import ModelBundle, get_active_bundle, bundle_files_ready
    try:
        from .selector_patch import patch_selector
    except ImportError:
        from selector_patch import patch_selector
    params = Params()
    if HARDWARE.get_device_type() != 'tizi' or params.get_bool('IsOnroad'):
        raise RuntimeError('installation requires an offroad c3x')
    manifest, report, source_bundle = verified_package(package, args.expected_step, args.expected_checkpoint_sha256)
    ref = source_bundle['ref']
    ui = iqroot / 'iqpilot/ui/layouts/settings/iq_panels.py'
    helper = Path(__file__).with_name('local_candidate.py')
    target = iqroot / 'iqpilot/selfdrive/iqmodeld/cd210_centerline_candidates_v2.py'
    compile(helper.read_text(), str(helper), 'exec')
    with installation_lock(root):
        if sha(ui) != args.expected_ui_sha256:
            raise RuntimeError('device UI changed since inspection')
        original = ui.read_text()
        patched = patch_selector(original)
        compile(patched, str(ui), 'exec')
        index = candidate_index(root, params.get('ModelManager_ModelsCache'), ref, args.expected_step)
        bundle = cached_bundle(source_bundle, index, args.expected_step, args.expected_checkpoint_sha256)
        ModelBundle(**bundle)
        backup = package / 'selector-install'
        receipt_path = backup / 'installation-receipt.json'
        active = Path('/data/params/d/ModelManager_ActiveBundle')
        before = active.read_bytes() if active.exists() else None
        copies = [(compiled / name, root / name) for name in report['artifacts']]
        copies += [(compiled / 'device-report.json', root / (ref + '.device-report.json')),
                   (package / 'payload.json', root / (ref + '.payload.json')),
                   (helper, target)]
        # Detect all existing-file conflicts before copying any model content.
        for source, destination in copies:
            if destination.exists() and sha(source) != sha(destination):
                raise RuntimeError(f'refusing to replace an existing candidate: {destination}')
        descriptor = root / (ref + '.bundle.json')
        encoded_bundle = (json.dumps(bundle, indent=2) + '\n').encode()
        if descriptor.exists() and descriptor.read_bytes() != encoded_bundle:
            raise RuntimeError('refusing to replace an existing candidate descriptor')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if not (receipt['candidate_ref'] == ref and receipt['source_payload_sha256'] == sha(package / 'payload.json')
                    and receipt['installed_ui_sha256'] == sha(ui) and descriptor.exists()
                    and all(destination.is_file() for _, destination in copies)):
                raise RuntimeError('existing installation receipt is inconsistent')
            print(json.dumps(dict(receipt, already_installed=True)), flush=True)
            return
        backup.mkdir(exist_ok=True)
        original_ui = backup / 'iq_panels.original.py'
        if original_ui.exists():
            saved_source = original_ui.read_text()
            if original not in (saved_source, patch_selector(saved_source)):
                raise RuntimeError('partial installation UI differs from saved original and patch')
        else:
            immutable_copy(ui, original_ui)
        original_active = backup / 'active-before.json'
        if not original_active.exists():
            with original_active.open('xb') as stream:
                stream.write(before or b'null')
        staged_descriptor = backup / (ref + '.bundle.json')
        if staged_descriptor.exists() and staged_descriptor.read_bytes() != encoded_bundle:
            raise RuntimeError('partial installation descriptor is inconsistent')
        if not staged_descriptor.exists():
            with staged_descriptor.open('xb') as stream:
                stream.write(encoded_bundle)
        for source, destination in copies:
            immutable_copy(source, destination)
        immutable_copy(staged_descriptor, descriptor)
        spec = importlib.util.spec_from_file_location('installed_centerline_candidates_v2', target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verified = module.validated_bundle(ref)
        if verified != bundle:
            raise RuntimeError('installed candidate differs from verified descriptor')
        class TestParams:
            def __init__(self): self.values = {'ModelManager_ActiveBundle': {'ref': 'prior'}}
            def get(self, key): return self.values.get(key)
            def remove(self, key): self.values.pop(key, None)
            def put(self, key, value): self.values[key] = value
        test_params = TestParams()
        module.select_candidate(test_params, offroad=True, ref=ref)
        decoded = get_active_bundle(test_params)
        if decoded.ref != ref or not bundle_files_ready(decoded):
            raise RuntimeError('native candidate decoding or file readiness failed')
        if params.get_bool('IsOnroad') or (active.read_bytes() if active.exists() else None) != before:
            raise RuntimeError('device or active model changed during installation')
        if sha(ui) != args.expected_ui_sha256:
            raise RuntimeError('device UI changed during installation')
        temporary = ui.with_name(ui.name + '.' + uuid4().hex + '.tmp')
        try:
            temporary.write_text(patched)
            os.chmod(temporary, ui.stat().st_mode)
            os.replace(temporary, ui)
        finally:
            temporary.unlink(missing_ok=True)
        receipt = dict(installed=True, candidate_ref=ref, display_name=bundle['displayName'], step=args.expected_step,
            selected_checkpoint_sha256=args.expected_checkpoint_sha256, candidate_index=index,
            picker_heading='Experimental', added_toggles=0, activated=False,
            active_before_sha256=hashlib.sha256(before or b'').hexdigest(),
            active_after_sha256=sha(active) if active.exists() else hashlib.sha256(b'').hexdigest(),
            original_ui_sha256=sha(original_ui), inspected_ui_sha256=args.expected_ui_sha256, installed_ui_sha256=sha(ui),
            helper_sha256=sha(target), helper_module=target.name, source_payload_sha256=sha(package / 'payload.json'),
            device_report_sha256=sha(compiled / 'device-report.json'), artifacts=report['artifacts'],
            native_bundle_decoded=True, files_ready=True,
            available_candidates=[item['ref'] for item in module.candidates()])
        with receipt_path.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
