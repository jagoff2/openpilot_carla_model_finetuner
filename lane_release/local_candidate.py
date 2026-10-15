"""Select immutable local CD210 candidates through the existing model picker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

# Keep the original public API usable by the already installed step-24 picker.
CANDIDATE_REF = "cd210-centerline-s24-55d9857fcc6f"
DISPLAY_NAME = "CD210 Centerline s24"
DESCRIPTOR_FILE = f"{CANDIDATE_REF}.bundle.json"
RECEIPT_FILE = f"{CANDIDATE_REF}.device-report.json"
_ACTIVE = "ModelManager_ActiveBundle"
_REF = re.compile(r"cd210-centerline-s([0-9]+)-([0-9a-f]{12})\Z")


def _root():
    from iqpilot.system.hardware.hw import Paths
    return Path(Paths.model_root())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(root, ref):
    match = _REF.fullmatch(ref) if isinstance(ref, str) else None
    if match is None:
        raise ValueError('invalid candidate reference')
    bundle = json.loads((root / f'{ref}.bundle.json').read_text())
    report = json.loads((root / f'{ref}.device-report.json').read_text())
    if not (bundle['ref'] == ref and bundle['runner'] == 'tinygrad'
            and report['runtime_ready'] is True and report['passed'] is True
            and len(bundle['models']) == 2
            and {m['type'] for m in bundle['models']} == {'vision', 'policy'}):
        raise ValueError('candidate is not runtime verified')
    payload_path = root / f'{ref}.payload.json'
    if payload_path.is_file():
        payload = json.loads(payload_path.read_text())
        checkpoint = payload['selected_checkpoint_sha256']
        if not (report['source_payload_sha256'] == _sha(payload_path)
                and isinstance(checkpoint, str)
                and re.fullmatch(r'[0-9a-f]{64}', checkpoint)
                and int(payload['step']) == int(match.group(1))
                and checkpoint.startswith(match.group(2))
                and payload['bundle']['ref'] == ref
                and report['frame_count'] >= 120):
            raise ValueError('candidate checkpoint identity failed')
    elif ref != CANDIDATE_REF:
        raise ValueError('candidate source payload is missing')
    return bundle, report


def candidates():
    """Read small descriptors when browsing, without loading or hashing models."""
    root = _root()
    entries = []
    for path in root.glob('cd210-centerline-s*.bundle.json'):
        ref = path.name.removesuffix('.bundle.json')
        try:
            bundle, _ = _identity(root, ref)
            step = int(_REF.fullmatch(ref).group(1))
            entries.append(dict(ref=ref, display_name=bundle.get('displayName') or f'CD210 Centerline s{step}',
                                short_name=f'CD210, checkpoint step {step}', step=step))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return sorted(entries, key=lambda entry: (entry['step'], entry['ref']), reverse=True)


def is_candidate_ref(ref):
    return isinstance(ref, str) and _REF.fullmatch(ref) is not None


def selected_ref(params):
    bundle = params.get(_ACTIVE)
    ref = bundle.get('ref') if isinstance(bundle, dict) else None
    return ref if is_candidate_ref(ref) else None


def validated_bundle(ref=CANDIDATE_REF):
    """Rehash artifacts on explicit selection; failed selection preserves Params."""
    root = _root()
    try:
        bundle, report = _identity(root, ref)
        for model in bundle['models']:
            for kind in ('artifact', 'metadata'):
                record = model[kind]
                name = record['fileName']
                if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
                    raise ValueError('invalid artifact path')
                expected = record['downloadUri']['sha256']
                if report['artifacts'][name] != expected or _sha(root / name) != expected:
                    raise ValueError(f'candidate integrity failed: {name}')
        return bundle
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f'{ref} is unavailable: {error}') from error


def select_candidate(params, *, offroad, ref=CANDIDATE_REF):
    if not offroad:
        raise RuntimeError('Change models while offroad')
    bundle = validated_bundle(ref)
    params.remove('ModelManager_DownloadIndex')
    params.remove('ModelManager_PendingIndex')
    params.remove('ModelRunnerTypeCache')
    params.put(_ACTIVE, bundle)
    return bundle


def is_selected(params, ref=CANDIDATE_REF):
    return selected_ref(params) == ref
