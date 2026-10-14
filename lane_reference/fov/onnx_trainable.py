"""Differentiable execution of the released, static supercombo ONNX graph.

The graph stays authoritative: no architecture or weights are guessed. Floating
weights have FP32 master copies and are cast at their use sites for native FP16
execution. Integer indices, masks and scalar arithmetic constants are buffers.
Export substitutes the trained weights into the original standard ONNX graph.
This intentionally supports only the operators present in the pinned checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
from torch import nn
from torch.nn import functional as F

DTYPES = {1: torch.float32, 10: torch.float16, 11: torch.float64,
          2: torch.uint8, 6: torch.int32, 7: torch.int64, 9: torch.bool}
OPS = {'Cast', 'Concat', 'Sub', 'Div', 'Conv', 'Gelu', 'Mul', 'Add',
       'ReduceMean', 'Relu', 'Sigmoid', 'Reshape', 'Gemm',
       'LayerNormalization', 'Unsqueeze', 'Transpose', 'GatherND', 'MatMul',
       'Slice', 'Squeeze', 'Where', 'Softmax', 'Gather'}


class ReleasedModel(nn.Module):
    def __init__(self, path: str | Path, precision: str = 'native'):
        super().__init__()
        if precision not in ('native', 'float32'):
            raise ValueError(precision)
        self.precision = precision
        self.source_path = str(Path(path).resolve())
        self.source_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.graph = onnx.load(path)
        onnx.checker.check_model(self.graph)
        unsupported = {n.op_type for n in self.graph.graph.node} - OPS
        if unsupported or any(n.domain for n in self.graph.graph.node):
            raise ValueError(f'Unsupported operators/domains: {unsupported}')
        self.input_names = [v.name for v in self.graph.graph.input]
        self.output_names = [v.name for v in self.graph.graph.output]
        self.weights = nn.ParameterDict()
        self.keys, self.original_dtypes, self.static = {}, {}, {}
        # Learnable initializers include affine weights/biases, layer scales and
        # positional embeddings. Keep arithmetic constants and masks immutable.
        consumers = {}
        for n in self.graph.graph.node:
            for i, name in enumerate(n.input):
                consumers.setdefault(name, []).append((n.op_type, i))
        self.initializer_roles = {}
        for j, init in enumerate(self.graph.graph.initializer):
            arr = numpy_helper.to_array(init).copy()
            tensor = torch.from_numpy(arr)
            key = f'w{j:04d}'
            self.keys[init.name] = key
            self.original_dtypes[init.name] = tensor.dtype
            trainable = arr.dtype.kind == 'f' and arr.size > 1 and np.isfinite(arr).all() and any(
                (op in ('Conv', 'Gemm', 'LayerNormalization') and i > 0) or
                op in ('MatMul', 'Add', 'Mul') for op, i in consumers.get(init.name, []))
            if trainable:
                self.weights[key] = nn.Parameter(tensor.float())
            else:
                self.register_buffer(key, tensor)
                self.static[init.name] = arr
            self.initializer_roles[init.name] = 'parameter' if trainable else 'constant'
        self.nodes = [(n.op_type, list(n.input), list(n.output),
                       {a.name: helper.get_attribute_value(a) for a in n.attribute}, n.name)
                      for n in self.graph.graph.node]
        self.uses = Counter(name for _, ins, _, _, _ in self.nodes for name in ins)

    def _ints(self, name):
        if name not in self.static:
            raise ValueError(f'Expected static integer control input: {name}')
        return self.static[name].reshape(-1).tolist()

    def _execute(self, op, x, a, names):
        if op == 'Cast':
            dtype = DTYPES[a['to']]
            if self.precision == 'float32' and dtype == torch.float16:
                dtype = torch.float32
            return x[0].to(dtype)
        if op == 'Concat': return torch.cat(x, dim=a['axis'])
        if op == 'Sub': return x[0] - x[1]
        if op == 'Div': return x[0] / x[1]
        if op == 'Mul': return x[0] * x[1]
        if op == 'Add': return x[0] + x[1]
        if op == 'MatMul': return x[0] @ x[1]
        if op == 'Relu': return F.relu(x[0])
        if op == 'Sigmoid': return torch.sigmoid(x[0])
        if op == 'Gelu': return F.gelu(x[0], approximate=a.get('approximate', b'none').decode())
        if op == 'Softmax': return F.softmax(x[0], dim=a.get('axis', -1))
        if op == 'Where': return torch.where(x[0], x[1], x[2])
        if op == 'Conv':
            if a.get('auto_pad', b'NOTSET') != b'NOTSET': raise ValueError('auto_pad')
            pads = a.get('pads', [0, 0, 0, 0])
            value = x[0]
            if pads[:2] != pads[2:]:
                value = F.pad(value, (pads[1], pads[3], pads[0], pads[2]))
                pads = [0, 0]
            return F.conv2d(value, x[1], x[2] if len(x) > 2 else None,
                            stride=a.get('strides', [1, 1]), padding=pads[:2],
                            dilation=a.get('dilations', [1, 1]), groups=a.get('group', 1))
        if op == 'ReduceMean':
            axes = self._ints(names[1]) if len(names) > 1 else a.get('axes')
            if axes == [] and a.get('noop_with_empty_axes', 0): return x[0]
            return x[0].mean(dim=tuple(axes) if axes else tuple(range(x[0].ndim)),
                             keepdim=bool(a.get('keepdims', 1)))
        if op == 'Reshape':
            shape = self._ints(names[1])
            if not a.get('allowzero', 0):
                shape = [x[0].shape[i] if d == 0 else d for i, d in enumerate(shape)]
            return x[0].reshape(shape)
        if op == 'Gemm':
            left = x[0].T if a.get('transA', 0) else x[0]
            right = x[1].T if a.get('transB', 0) else x[1]
            # addmm preserves one rounded output for GEMM plus its bias.
            if len(x) > 2:
                return torch.addmm(x[2], left, right, beta=a.get('beta', 1.), alpha=a.get('alpha', 1.))
            return a.get('alpha', 1.) * (left @ right)
        if op == 'LayerNormalization':
            axis = a.get('axis', -1) % x[0].ndim
            if a.get('stash_type', 1) != 1: raise ValueError('LayerNorm stash_type')
            # ONNX casts normalized values to input dtype before affine scale.
            # This differs subtly from torch.layer_norm's fused FP16 affine.
            dims = tuple(range(axis, x[0].ndim))
            value = x[0].float()
            mean = value.mean(dims, keepdim=True)
            variance = ((value - mean) ** 2).mean(dims, keepdim=True)
            normalized = ((value - mean) * torch.rsqrt(variance + a.get('epsilon', 1e-5))).to(x[0].dtype)
            result = normalized * x[1]
            if len(x) > 2: result = result + x[2]
            return result
        if op == 'Transpose': return x[0].permute(a.get('perm', list(reversed(range(x[0].ndim)))))
        if op == 'Unsqueeze':
            value = x[0]
            axes = self._ints(names[1])
            for axis in sorted(d % (value.ndim + len(axes)) for d in axes):
                value = value.unsqueeze(axis)
            return value
        if op == 'Squeeze':
            value = x[0]
            if len(names) == 1: return value.squeeze()
            for axis in sorted((d % value.ndim for d in self._ints(names[1])), reverse=True):
                if value.shape[axis] != 1: raise ValueError('Squeeze dimension is not one')
                value = value.squeeze(axis)
            return value
        if op == 'Slice':
            starts, ends = self._ints(names[1]), self._ints(names[2])
            axes = self._ints(names[3]) if len(names) > 3 else list(range(len(starts)))
            steps = self._ints(names[4]) if len(names) > 4 else [1] * len(starts)
            selectors = [slice(None)] * x[0].ndim
            for start, end, axis, step in zip(starts, ends, axes, steps):
                if step <= 0: raise ValueError('Nonpositive slice steps not implemented')
                selectors[axis] = slice(start, end, step)
            return x[0][tuple(selectors)]
        if op == 'Gather':
            axis = a.get('axis', 0) % x[0].ndim
            idx = x[1].long()
            idx = torch.where(idx < 0, idx + x[0].shape[axis], idx)
            shape = tuple(x[0].shape[:axis]) + tuple(idx.shape) + tuple(x[0].shape[axis + 1:])
            return torch.index_select(x[0], axis, idx.reshape(-1)).reshape(shape)
        if op == 'GatherND':
            if a.get('batch_dims', 0): raise ValueError('GatherND batch_dims')
            return x[0][tuple(x[1].long().unbind(-1))]
        raise NotImplementedError(op)

    def forward(self, inputs: dict[str, torch.Tensor], intermediates: bool = False):
        env = dict(inputs)
        for name, key in self.keys.items():
            value = self.weights[key] if key in self.weights else getattr(self, key)
            dtype = self.original_dtypes[name]
            if self.precision == 'float32' and dtype == torch.float16: dtype = torch.float32
            env[name] = value.to(dtype)
        if self.precision == 'float32':
            env = {k: v.float() if v.dtype == torch.float16 else v for k, v in env.items()}
        remaining = self.uses.copy()
        for op, ins, outs, attrs, name in self.nodes:
            try:
                result = self._execute(op, [env[i] for i in ins if i], attrs, ins)
                if len(outs) != 1: raise ValueError(f'Multiple outputs: {outs}')
                env[outs[0]] = result
            except Exception as exc:
                raise RuntimeError(f'{name} ({op}), inputs {ins}: {exc}') from exc
            if not intermediates:
                for i in ins:
                    remaining[i] -= 1
                    if remaining[i] == 0 and i not in self.output_names:
                        env.pop(i, None)
        return env if intermediates else env[self.output_names[0]]

    def export(self, path: str | Path):
        if Path(path).resolve() == Path(self.source_path).resolve():
            raise ValueError('Export must not overwrite the source checkpoint')
        model = copy.deepcopy(self.graph)
        for init in model.graph.initializer:
            key = self.keys[init.name]
            if key in self.weights:
                value = self.weights[key].detach().cpu().to(self.original_dtypes[init.name]).numpy()
                init.CopyFrom(numpy_helper.from_array(value, init.name))
        onnx.checker.check_model(model)
        onnx.save(model, path)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def parameter_inventory(self):
        return [{'name': name, 'key': key, 'shape': list(self.weights[key].shape),
                 'elements': self.weights[key].numel()}
                for name, key in self.keys.items() if key in self.weights]
