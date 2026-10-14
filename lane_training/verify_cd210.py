"""CPU numerical and export verification of actual CD210 checkpoint weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

from lane_dataset.loader import SequenceEpisode
from .cd210 import DEFAULT_CHECKPOINT
from .cd210_model import create_model, export_onnx, save_checkpoint, load_checkpoint, verify_exports
from .smoke import parameter_digest


def float32_graph(graph):
    """Independent ORT reference with the same explicit FP32 training precision."""
    graph = copy.deepcopy(graph)
    for value in list(graph.graph.input)+list(graph.graph.output)+list(graph.graph.value_info):
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    for initializer in graph.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT16:
            initializer.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float32), initializer.name))
    for node in graph.graph.node:
        if node.op_type == 'Cast':
            for attribute in node.attribute:
                if attribute.name == 'to' and attribute.i == onnx.TensorProto.FLOAT16:
                    attribute.i = onnx.TensorProto.FLOAT
    onnx.checker.check_model(graph)
    return graph


def runtime(graph):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(graph.SerializeToString(), options, providers=['CPUExecutionProvider'])


def compare(actual, expected):
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    return dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
        allclose=bool(np.allclose(actual,expected,atol=2e-4,rtol=2e-4)), atol=2e-4, rtol=2e-4)


def verify(manifest, report_path, *, checkpoint=DEFAULT_CHECKPOINT):
    start = time.perf_counter()
    torch.set_num_threads(2)
    model = create_model(checkpoint).eval()
    before = parameter_digest(model)
    sessions = {role: runtime(float32_graph(getattr(model,role).graph)) for role in ('vision','policy')}
    native_sessions = {role: runtime(getattr(model,role).graph) for role in ('vision','policy')}
    comparisons = []
    recorded = []
    generator = torch.Generator().manual_seed(210)
    with SequenceEpisode(manifest) as episode:
        indices = [0, int(episode.labels['label_indices'][0]), len(episode)//2, len(episode)-1]
        for index in indices:
            first, current = episode.frame(max(0,index-4)), episode.frame(index)
            cameras = {name: torch.from_numpy(np.concatenate([first[k],current[k]])[None])
                for k,name in enumerate(('img','big_img'))}
            with torch.no_grad():
                vision = model.vision(cameras)
            policy_inputs = dict(features_buffer=vision[:,None,1064:].expand(1,25,512).clone(),
                desire_pulse=torch.zeros(1,25,8), traffic_convention=torch.tensor([[1.,0.]]))
            policy_inputs['features_buffer'] += torch.randn(1,25,512,generator=generator)*.01
            if index == len(episode)//2:
                policy_inputs['desire_pulse'][0,-1,1] = 1
                policy_inputs['traffic_convention'] = torch.tensor([[0.,1.]])
            for role, inputs in (('vision',cameras),('policy',policy_inputs)):
                with torch.no_grad():
                    actual = getattr(model,role)(inputs).numpy()
                expected = sessions[role].run(None, {k:v.numpy() for k,v in inputs.items()})[0]
                native = native_sessions[role].run(None, {k:v.numpy().astype(np.float16) if v.is_floating_point()
                    else v.numpy() for k,v in inputs.items()})[0]
                row = dict(frame=index, role=role, float32_onnx=compare(actual,expected),
                    original_fp16_onnx_precision_difference=compare(actual,native))
                comparisons.append(row)
                print(json.dumps(row), flush=True)
            recorded.append(dict(**cameras, desire_pulse=policy_inputs['desire_pulse'],
                traffic_convention=policy_inputs['traffic_convention'], features_buffer=policy_inputs['features_buffer'][:,:24]))
        manifest_sha = hashlib.sha256(episode.path.read_bytes()).hexdigest()

    # Directional derivative through both complete graphs, at unchanged weights.
    model.zero_grad(set_to_none=True)
    output = model(recorded[1])
    scalar = output[0,:495].float().square().mean()
    scalar.backward()
    gradient_checks = {}
    for role in ('vision','policy'):
        parameters = [(name,p) for name,p in getattr(model,role).named_parameters() if p.grad is not None]
        name, parameter = max(parameters, key=lambda item: float(item[1].grad.abs().max()))
        index = int(parameter.grad.abs().flatten().argmax())
        analytic = float(parameter.grad.flatten()[index])
        original = parameter.detach().flatten()[index].clone()
        epsilon = max(1e-4,abs(float(original))*1e-3)
        values = []
        with torch.no_grad():
            for sign in (1,-1):
                parameter.flatten()[index] = original+sign*epsilon
                values.append(float(model(recorded[1])[0,:495].float().square().mean()))
            parameter.flatten()[index] = original
        numeric = (values[0]-values[1])/(2*epsilon)
        relative = abs(numeric-analytic)/max(1e-5,abs(numeric),abs(analytic))
        gradient_checks[role] = dict(parameter=name, element=index, analytic=analytic, numeric=numeric,
            epsilon=epsilon, relative_error=relative, passed=relative < .05,
            gradient_tensors=len(parameters), nonzero_elements=sum(int(p.grad.count_nonzero()) for _,p in parameters),
            all_finite=all(bool(torch.isfinite(p.grad).all()) for _,p in parameters))

    output_dir = Path(report_path).parent/'cd210-export-verification'
    checkpoint_path = output_dir/'cd210-initial.pt'
    checkpoint_sha = save_checkpoint(model, checkpoint_path)
    restored, package = load_checkpoint(checkpoint_path)
    with torch.no_grad():
        expected = model(recorded[1])
        reloaded = restored(recorded[1])
    state_roundtrip = parameter_digest(restored) == before and bool(torch.equal(expected,reloaded))
    exports = export_onnx(model, output_dir/'onnx')
    export_roundtrip = {}
    for role,binding in exports.items():
        emitted = onnx.load(binding['path'])
        source = getattr(model,role).graph
        export_roundtrip[role] = bool(all(a.name == b.name and np.array_equal(numpy_helper.to_array(a),numpy_helper.to_array(b))
            for a,b in zip(source.graph.initializer,emitted.graph.initializer)) and source.graph.node == emitted.graph.node)
    after = parameter_digest(model)
    report = dict(scope='actual-CD210-source-ONNX-differentiable-export', checkpoint_binding=verify_exports(checkpoint),
        device='cpu', ort_providers=['CPUExecutionProvider'], onnxruntime_version=ort.__version__,
        torch_version=torch.__version__, training_precision='float32 masters and arithmetic',
        manifest=str(Path(manifest).resolve()), manifest_sha256=manifest_sha,
        numerical_comparisons=comparisons, gradient_checks=gradient_checks,
        parameter_count=sum(p.numel() for p in model.parameters()), parameters_unchanged=before==after,
        parameter_sha256=before, optimizer_steps=0, checkpoint_path=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_sha,
        checkpoint_roundtrip_passed=state_roundtrip, onnx_exports=exports, onnx_roundtrip=export_roundtrip,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        elapsed_seconds=time.perf_counter()-start)
    report['passed'] = bool(all(row['float32_onnx']['allclose'] for row in comparisons)
        and all(row['passed'] and row['all_finite'] and row['nonzero_elements'] > 0 for row in gradient_checks.values())
        and before==after and state_roundtrip and all(export_roundtrip.values()))
    report['scope_limit'] = 'Official source ONNX parity; compiled QCOM runtime parity and driving behavior are not evaluated here.'
    Path(report_path).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoint_binding','source_hashes','numerical_comparisons')},indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    return 0 if verify(args.manifest,args.report)['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
