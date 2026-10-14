"""Trainable CD210 vision and policy loaded from the original pinned ONNX exports."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import pickle

import onnx
import torch

from lane_training.cd210 import CD210SplitModel, DEFAULT_CHECKPOINT, verify_checkpoint
from lane_training.fetch_cd210 import EXPORT_DIR, EXPORTS, COMMIT, SOURCE, verify_file
from lane_training.onnx_model import ReleasedModel


def verify_exports(bundle_path=DEFAULT_CHECKPOINT, directory=EXPORT_DIR):
    binding = verify_checkpoint(bundle_path)
    exports = {}
    for role, pinned in EXPORTS.items():
        path = Path(directory)/pinned['file']
        verify_file(path, pinned)
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        metadata = {item.key: item.value for item in graph.metadata_props}
        # The source files have already matched exact upstream Git LFS SHA-256s.
        slices = pickle.loads(base64.b64decode(metadata['output_slices']))
        if metadata['model_checkpoint'] != binding['source_checkpoints'][role] or metadata['model_checkpoint'] != pinned['checkpoint']:
            raise ValueError(f'{role} checkpoint does not match the selected CD210 bundle')
        def shapes(values):
            return {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in values}
        if shapes(graph.graph.input) != binding['input_shapes'][role] or shapes(graph.graph.output) != binding['output_shapes'][role]:
            raise ValueError(f'{role} ONNX interface differs from the selected CD210 bundle')
        compiled_metadata = Path(bundle_path).parent/f'driving_{role}_c210m_metadata.pkl'
        if slices != pickle.loads(compiled_metadata.read_bytes())['output_slices']:
            raise ValueError(f'{role} ONNX output slices differ from the selected CD210 bundle')
        exports[role] = dict(**pinned, path=str(path.resolve()), nodes=len(graph.graph.node),
            inputs={v.name: v.type.tensor_type.elem_type for v in graph.graph.input},
            outputs={v.name: v.type.tensor_type.elem_type for v in graph.graph.output})
    return dict(compiled_bundle=binding, upstream_commit=COMMIT, source=SOURCE, exports=exports,
        training_graph_status='original_CD210_ONNX_exports_hash_and_metadata_verified', passed=True)


def create_model(bundle_path=DEFAULT_CHECKPOINT, device='cpu', precision='float32', *, directory=EXPORT_DIR):
    binding = verify_exports(bundle_path, directory)
    modules = {role: ReleasedModel(Path(directory)/entry['file'], precision=precision,
        expected_sha256=entry['sha256']) for role, entry in EXPORTS.items()}
    model = CD210SplitModel(modules['vision'], modules['policy']).to(device)
    model.checkpoint_binding = binding
    return model


def save_checkpoint(model, path, *, optimizer=None, step=0, training_state=None):
    """Save weights and provenance without overwriting either original ONNX file."""
    destination = Path(path).resolve()
    if destination.suffix != '.pt':
        raise ValueError('training checkpoint must use .pt')
    package = dict(format='cd210-centerline-v1', upstream_commit=COMMIT,
        source_hashes={k:v['sha256'] for k,v in EXPORTS.items()}, step=int(step),
        model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=optimizer.state_dict() if optimizer else None,
        training_state=training_state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.pt.partial')
    torch.save(package, partial)
    partial.replace(destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_checkpoint(path, *, bundle_path=DEFAULT_CHECKPOINT, device='cpu', precision='float32', directory=EXPORT_DIR):
    package = torch.load(path, map_location='cpu', weights_only=True)
    if (package.get('format') != 'cd210-centerline-v1' or package.get('upstream_commit') != COMMIT
            or package.get('source_hashes') != {k:v['sha256'] for k,v in EXPORTS.items()}):
        raise ValueError('training checkpoint is not bound to the selected CD210 exports')
    model = create_model(bundle_path, device=device, precision=precision, directory=directory)
    model.load_state_dict(package['model'], strict=True)
    return model, package


def export_onnx(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A durable exclusive claim also prevents two exports racing in one folder.
    names = ['export.json', *(entry['file'] for entry in EXPORTS.values())]
    if any((directory/name).exists() for name in names):
        raise FileExistsError('model export already exists; choose a new directory')
    with (directory/'.model-export-claimed').open('x', encoding='utf-8') as stream:
        stream.write('This export directory must never be reused.\n')
    bindings = {role: dict(path=str(directory/entry['file']),
        sha256=getattr(model,role).export(directory/entry['file'])) for role,entry in EXPORTS.items()}
    (directory/'export.json').write_text(json.dumps(dict(model='CD210', upstream_commit=COMMIT,
        source_hashes={k:v['sha256'] for k,v in EXPORTS.items()}, exports=bindings), indent=2)+'\n', encoding='utf-8')
    return bindings
