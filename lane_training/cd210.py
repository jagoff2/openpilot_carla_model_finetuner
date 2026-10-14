"""Pinned CD210 bundle, policy output adapter and separate vision/policy rollout.

The bundle pins original QCOM hashes and exact metadata. Training requires only
the metadata and original ONNX exports, loaded by cd210_model.py. Native PKLs are
compiled on the target device during export.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from lane_dataset.config import PROJECT
from .adapters import FlatPlanAdapter

DEFAULT_CHECKPOINT = PROJECT/'models/cd210/bundle/bundle.json'
PINNED_FILES = {
    'driving_policy_c210m_tinygrad.pkl': '6bcf9668455c022ccaa6443ed2daddc5b4348ffbdf68dd5abd9c37a1b17bfe98',
    'driving_policy_c210m_metadata.pkl': '15c8c1ad9073424ee1101b1f7170421140ed308ebaa7c917001130fb1760420b',
    'driving_vision_c210m_tinygrad.pkl': '10fd116056fd1790553c20e09b2b640ebcb85e72418fc9685e0800a513bbe950',
    'driving_vision_c210m_metadata.pkl': 'a2be39088d38550e818f5ac1c6300a64605ba0bd0676a27fe7bbea9fddae90a1',
}
POLICY_PLAN = FlatPlanAdapter(1000, 0, 33, 15, yaw_column=11)
# Training wrapper appends current vision features to the native policy output.
ROLLOUT_PLAN = FlatPlanAdapter(1512, 0, 33, 15, yaw_column=11)


def verify_checkpoint(bundle_path=DEFAULT_CHECKPOINT):
    path = Path(bundle_path).resolve()
    bundle = json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('internalName') != 'C210M' or not bundle.get('is20hz'):
        raise ValueError('expected the selected CD210/C210M 20 Hz checkpoint')
    bindings = {}
    metadata = {}
    for role in bundle['models']:
        for kind in ('artifact', 'metadata'):
            entry = role[kind]
            name = entry['fileName']
            if name not in PINNED_FILES or entry['downloadUri']['sha256'] != PINNED_FILES[name]:
                raise ValueError('CD210 artifact binding differs from the pinned checkpoint')
            file = path.parent/name
            if kind=='artifact' and not file.exists():
                bindings[name]=dict(path=str(file),sha256=PINNED_FILES[name],bytes=None,verified=False,
                    required_for_training=False)
                continue
            content = file.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != PINNED_FILES[name]:
                raise ValueError(f'CD210 artifact hash mismatch: {name}')
            bindings[name] = dict(path=str(file), sha256=digest, bytes=len(content),verified=True)
            if kind == 'metadata':
                # Only parse exact hash-verified metadata. Never unpickle QCOM executables.
                metadata[role['type']] = pickle.loads(content)
    if set(bindings) != set(PINNED_FILES):
        raise ValueError('incomplete CD210 bundle')
    policy, vision = metadata['policy'], metadata['vision']
    if (policy['input_shapes'] != {'desire_pulse': (1,25,8), 'traffic_convention': (1,2), 'features_buffer': (1,25,512)}
            or policy['output_shapes'] != {'outputs': (1,1000)}
            or policy['output_slices']['plan'] != slice(0,990)
            or vision['input_shapes'] != {'img': (1,12,128,256), 'big_img': (1,12,128,256)}
            or vision['output_shapes'] != {'outputs': (1,1576)}
            or vision['output_slices']['hidden_state'] != slice(1064,1576)):
        raise ValueError('CD210 metadata does not match the training adapter')
    return dict(model='CD210', internal_name='C210M', checkpoint=str(path),
        bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), files=bindings,
        source_checkpoints={role: value['model_checkpoint'] for role,value in metadata.items()},
        input_shapes={role: value['input_shapes'] for role,value in metadata.items()},
        output_shapes={role: value['output_shapes'] for role,value in metadata.items()},
        plan_mean_slice=[0,495], plan_shape=[33,15], vision_feature_slice=[1064,1576],
        compiled_artifacts_verified=all(v['verified'] for k,v in bindings.items() if k.endswith('_tinygrad.pkl')),
        training_graph_status='pinned_native_metadata; see cd210_model.verify_exports for source graphs', passed=True)


class CD210SplitModel(torch.nn.Module):
    """Compose supplied differentiable CD210 modules, preserving current vision gradients."""
    def __init__(self, vision, policy):
        super().__init__()
        self.vision, self.policy = vision, policy

    def forward(self, inputs):
        if set(inputs) != {'img', 'big_img', 'desire_pulse', 'traffic_convention', 'features_buffer'}:
            raise ValueError('CD210 training wrapper requires exactly five inputs')
        if inputs['features_buffer'].shape != (1,24,512):
            raise ValueError('wrapper history contains the 24 prior 5 Hz features')
        vision = self.vision({name: inputs[name] for name in ('img', 'big_img')})
        if vision.shape != (1,1576):
            raise ValueError('CD210 vision output must be [1,1576]')
        current = vision[:,1064:1576]
        features = torch.cat([inputs['features_buffer'], current[:,None,:]], dim=1)
        policy = self.policy(dict(features_buffer=features, desire_pulse=inputs['desire_pulse'],
            traffic_convention=inputs['traffic_convention']))
        if policy.shape != (1,1000):
            raise ValueError('CD210 policy output must be [1,1000]')
        return torch.cat([policy, current], dim=1)


def student_rollout(episode, student, *, device='cpu', auxiliary_dtype=None,
                    detach_history=True, stop_after_last_label=True):
    """20 Hz images; policy features t-96,t-92,...,t-4,current vision.

    The native CD210 policy sees 25 entries including current vision. The
    wrapper takes only 24 previous entries, then computes/appends its own
    current feature. No recorded commands, speed or teacher features are inputs.
    """
    dtype = auxiliary_dtype or torch.float32
    zero = torch.zeros(512, device=device, dtype=dtype)
    features = deque([zero]*96, maxlen=96)
    previous = zero
    camera = deque([np.zeros((2,6,128,256), np.uint8)]*5, maxlen=5)
    desires = np.zeros((100,8), np.float32)
    previous_desire = np.zeros(8, np.float32)
    selected = set(int(i) for i in episode.labels['label_indices'])
    last = max(selected) if stop_after_last_label else len(episode)-1
    predictions = []
    for index in range(last+1):
        frame = episode.frame(index)
        if frame.shape != (2,6,128,256) or frame.dtype != np.uint8:
            raise ValueError('CD210 requires stock packed camera frames')
        camera.append(frame)
        metadata = episode.frames[index]
        desire = np.asarray(metadata['desire'], np.float32).copy()
        desire[0] = 0
        pulse = np.where(desire-previous_desire > .99, desire, 0)
        previous_desire = desire
        desires[:-1] = desires[1:]
        desires[-1] = pulse
        features.append(previous)
        inputs = {name: torch.as_tensor(np.concatenate([camera[0][cam], camera[-1][cam]])[None],
            device=device) for cam,name in enumerate(('img','big_img'))}
        inputs.update(desire_pulse=torch.as_tensor(desires.reshape(25,4,8).max(1)[None], device=device, dtype=dtype),
            traffic_convention=torch.as_tensor([metadata['traffic_convention']], device=device, dtype=dtype),
            features_buffer=torch.stack(list(features)[::4])[None])
        with torch.set_grad_enabled(not detach_history or index in selected):
            output = student(inputs)
        if output.shape != (1,1512) or not bool(torch.isfinite(output[:,1000:]).all()):
            raise ValueError('invalid CD210 training-wrapper output or vision features')
        previous = output[0,1000:].detach() if detach_history else output[0,1000:]
        if index in selected:
            predictions.append(output[0])
    return torch.stack(predictions), episode.selected_targets()


def backward(episode, student, reference, **kwargs):
    from .rollout import centerline_backward
    return centerline_backward(episode, student, reference, plan_adapter=ROLLOUT_PLAN,
        rollout=student_rollout, **kwargs)
