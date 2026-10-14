"""Fetch the immutable upstream CD210 ONNX exports and verify Git LFS hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import shutil

COMMIT = '5b6436a90cf6902b8aaa71c2b6f3d7164d8ae391'
SOURCE = f'https://github.com/commaai/openpilot/commit/{COMMIT}'
EXPORT_DIR = Path(__file__).resolve().parents[1]/'models/cd210'
EXPORTS = {
    'policy': dict(file='driving_policy.onnx', sha256='78477124cbf3ffe30fa951ebada8410b43c4242c6054584d656f1d329b067e15', bytes=14060847,
                   checkpoint='a27b3122-733e-4a65-938b-acfebebbe5e8/100'),
    'vision': dict(file='driving_vision.onnx', sha256='ee29ee5bce84d1ce23e9ff381280de9b4e4d96d2934cd751740354884e112c66', bytes=46877473,
                   checkpoint='6a7d09ad-bcc9-43bc-916d-29287e60cee2/200'),
}


def verify_file(path, binding):
    path = Path(path)
    if path.stat().st_size != binding['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
        raise ValueError(f'CD210 source export failed hash verification: {path}')


def fetch(directory=EXPORT_DIR,source_exports=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(model='CD210', upstream_commit=COMMIT, source=SOURCE, exports={})
    for role, binding in EXPORTS.items():
        path = directory/binding['file']
        if source_exports is not None and not path.exists():
            source=Path(source_exports)/binding['file'];verify_file(source,binding)
            shutil.copyfile(source,path)
        pointer_url = f'https://raw.githubusercontent.com/commaai/openpilot/{COMMIT}/selfdrive/modeld/models/{path.name}'
        if not path.exists():
            with urllib.request.urlopen(pointer_url, timeout=30) as response:
                pointer = response.read().decode('ascii')
            expected = f"version https://git-lfs.github.com/spec/v1\noid sha256:{binding['sha256']}\nsize {binding['bytes']}\n"
            if pointer != expected:
                raise ValueError('upstream CD210 LFS pointer differs from the pinned release')
            batch_url = 'https://gitlab.com/commaai/openpilot-lfs.git/info/lfs/objects/batch'
            request = urllib.request.Request(batch_url, data=json.dumps(dict(operation='download', transfers=['basic'],
                objects=[dict(oid=binding['sha256'], size=binding['bytes'])])).encode(),
                headers={'Content-Type': 'application/vnd.git-lfs+json', 'Accept': 'application/vnd.git-lfs+json'})
            with urllib.request.urlopen(request, timeout=30) as response:
                obj = json.load(response)['objects'][0]
            action = obj['actions']['download']
            request = urllib.request.Request(action['href'], headers=action.get('header', {}))
            partial = path.with_suffix('.onnx.partial')
            with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as stream:
                while chunk := response.read(1024*1024):
                    stream.write(chunk)
            verify_file(partial, binding)
            partial.replace(path)
        verify_file(path, binding)
        manifest['exports'][role] = dict(**binding, path=str(path.resolve()), pointer_url=pointer_url)
        print(json.dumps(dict(stage='verified_export', role=role, **manifest['exports'][role])), flush=True)
    (directory/'source.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


def fetch_bundle(directory=EXPORT_DIR/'bundle',source_bundle=None):
    """Recreate exact pinned metadata from public ONNX, without old QCOM downloads."""
    import base64
    import pickle
    import onnx
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    template=Path(__file__).resolve().parents[1]/'pipeline/assets/cd210-bundle.json'
    bundle=json.loads(template.read_text())
    for role in bundle['models']:
        binding=EXPORTS[role['type']];source=directory.parent/binding['file'];verify_file(source,binding)
        graph=onnx.load(source);props={p.key:p.value for p in graph.metadata_props}
        shapes=lambda values:{v.name:tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in values}
        metadata=dict(model_checkpoint=props['model_checkpoint'],output_slices=pickle.loads(base64.b64decode(props['output_slices'])),
            input_shapes=shapes(graph.graph.input),output_shapes=shapes(graph.graph.output))
        for kind in ('artifact','metadata'):
            entry=role[kind];path=directory/entry['fileName'];expected=entry['downloadUri']['sha256']
            if kind=='artifact' and source_bundle is None:continue
            if not path.exists():
                temporary=path.with_suffix('.pkl.partial')
                if source_bundle is not None:shutil.copyfile(Path(source_bundle)/path.name,temporary)
                else:temporary.write_bytes(pickle.dumps(metadata,protocol=4))
                if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:
                    raise ValueError(f'CD210 bundle hash mismatch: {path.name}')
                temporary.replace(path)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'CD210 bundle hash mismatch: {path.name}')
            print(json.dumps(dict(stage='verified_bundle_file',file=path.name,sha256=expected)),flush=True)
    (directory/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=EXPORT_DIR)
    parser.add_argument('--source-exports',type=Path,help='Use an existing hash-verified ONNX cache instead of downloading')
    parser.add_argument('--source-bundle',type=Path,help='Use an existing matching CD210 bundle directory')
    args = parser.parse_args()
    fetch(args.directory,args.source_exports)
    fetch_bundle(args.directory/'bundle',args.source_bundle)


if __name__ == '__main__':
    main()
