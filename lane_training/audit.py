"""CPU objective/sequence audit on accepted data, using an explicit synthetic model.

This verifies geometry targets and the CD210 adapter contract. It does not run
CD210 checkpoint inference, change checkpoint weights or qualify driving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from lane_dataset.adapter import PLAN_TIMES
from lane_dataset.loader import SequenceEpisode
from .cd210 import DEFAULT_CHECKPOINT, CD210SplitModel, backward, verify_checkpoint
from .centerline import centerline_loss
from .targets import episode_corridors


class FixtureVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.001))
    def forward(self, inputs):
        observed = (inputs['img'][:,6:].float().mean()+inputs['big_img'][:,6:].float().mean())/255
        return (observed*self.scale).expand(1,1576)


class FixturePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(.25))
    def forward(self, inputs):
        plan = torch.zeros(1,33,15)
        plan[0,:,0] = torch.linspace(0,20,33)
        plan[0,:,1] = self.offset+inputs['features_buffer'][0,-1,0]
        return torch.cat([plan.reshape(1,495), torch.zeros(1,505)], dim=1)


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def audit(manifest, checkpoint=DEFAULT_CHECKPOINT, progress=None):
    torch.set_num_threads(2)
    started = time.perf_counter()
    binding = verify_checkpoint(checkpoint)
    student = CD210SplitModel(FixtureVision(), FixturePolicy())
    reference = CD210SplitModel(FixtureVision(), FixturePolicy()).requires_grad_(False)
    before = parameter_hash(student)
    with SequenceEpisode(manifest) as episode:
        corridors = episode_corridors(episode)
        coverage = []
        comparisons = []
        for row, corridor in enumerate(corridors):
            horizon = episode.labels['plan_mask'][row,:,:3].all(1)
            coverage.append(dict(label_index=corridor.label_index,
                observed_forward_m=float(corridor.s[-1]-corridor.ego_s),
                old_trajectory_horizon_s=float(PLAN_TIMES[horizon].max()) if horizon.any() else 0.))
            distance = min(20., float(corridor.s[-1]-corridor.ego_s))
            q = corridor.ego_s+np.linspace(0,distance,33)
            center, tangent, _, _, mask = corridor.sample(q)
            if not mask.all():
                continue
            good = torch.tensor(center, dtype=torch.float32)[None]
            normal = np.c_[-tangent[:,1],tangent[:,0]]
            shifted = torch.tensor(center+.4*normal, dtype=torch.float32)[None].requires_grad_()
            good_loss, _ = centerline_loss(good, good, [corridor])
            bad_loss, info = centerline_loss(shifted, good, [corridor])
            bad_loss.backward()
            comparisons.append(dict(label_index=corridor.label_index, centered_loss=float(good_loss),
                shifted_loss=float(bad_loss.detach()),
                mean_abs_lateral_m=info['mean_abs_lateral_m'],
                finite_path_gradient=bool(torch.isfinite(shifted.grad).all()),
                centered_preferred=bool(good_loss < bad_loss)))
        prediction, diagnostics = backward(episode, student, reference, device='cpu',
            auxiliary_dtype=torch.float32, complete_sequence=True, progress=progress)
        manifest_path = episode.path
        frame_count = len(episode)
    gradients = [p.grad for p in student.parameters() if p.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
    nonzero = sum(int(g.count_nonzero()) for g in gradients)
    after = parameter_hash(student)
    report = dict(schema_version=1, scope='accepted-data-objective-and-CD210-interface-with-synthetic-model',
        checkpoint_binding=binding, checkpoint_inference_performed=False,
        model_kind='small synthetic vision/policy contract fixture; no CD210 model parameters loaded',
        manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        episode_file_hashes_verified=True, device='cpu', collection_gpus_used=False,
        dense_frame_count=frame_count, gradients_finite=finite, nonzero_gradient_elements=nonzero,
        fixture_parameters_unchanged=before==after, optimizer_steps=0,
        prediction_finite=bool(torch.isfinite(prediction).all()),
        coverage=coverage, geometry_comparisons=comparisons, rollout=diagnostics,
        elapsed_seconds=time.perf_counter()-started,
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    report['passed'] = bool(finite and nonzero and before==after and report['prediction_finite']
        and comparisons and all(r['centered_preferred'] and r['finite_path_gradient'] for r in comparisons)
        and diagnostics['backward_calls']==len(corridors) and diagnostics['student_ticks']==frame_count
        and diagnostics['reference_ticks']==frame_count and not diagnostics['reference_has_gradients'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.checkpoint,
        progress=lambda stage,tick,total: print(json.dumps(dict(stage=stage,tick=tick,total=total)),flush=True))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('passed','scope','dense_frame_count','gradients_finite',
        'nonzero_gradient_elements','fixture_parameters_unchanged','optimizer_steps','elapsed_seconds')},indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
