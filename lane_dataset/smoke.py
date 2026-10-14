"""Opt-in real-checkpoint forward/backward check, never an optimizer step.

With --manifest this traverses the accepted dense episode through the actual
student. --diagnostic-manifest explicitly inspects an unaccepted recording with
identical integrity checks. Without either option the report identifies a
synthetic contract fixture. Diagnostics/fixtures cannot qualify a corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from .adapter import expert_trajectory_loss
from .preprocessing import DEPENDENCY_ROOT, RebelliousHopeState


def recorded_rollout_backward(episode, model, *, device="cpu", auxiliary_dtype=None, progress=None):
    """Accumulate the same detached-history loss gradient with one label graph.

    This smoke path releases each labelled graph immediately. It traverses every
    frame with the current student, including the suffix, without an optimizer.
    """
    import torch
    from .loader import student_rollout
    targets = episode.selected_targets()
    kwargs = {name: torch.as_tensor(targets[name], device=device)
              for name in ("plan_targets", "plan_mask", "lateral_targets", "lateral_mask")}
    valid_rows = kwargs["plan_mask"].any(dim=(1,2)) | kwargs["lateral_mask"]
    normalizer = max(1,int(valid_rows.sum()))
    label_rows = {int(index): row for row,index in enumerate(targets["label_indices"])}
    tick, backwards = 0, 0
    model.zero_grad(set_to_none=True)
    def forward_and_backward(inputs):
        nonlocal tick, backwards
        output = model(inputs)
        row = label_rows.get(tick)
        if row is not None and bool(valid_rows[row]):
            selected = {name:value[row:row+1] for name,value in kwargs.items()}
            loss, _ = expert_trajectory_loss(output.float(), **selected)
            (loss / normalizer).backward()
            backwards += 1
        tick += 1
        if progress is not None and (tick % 100 == 0 or tick == len(episode)):
            progress(tick,len(episode))
        return output.detach()
    prediction, returned_targets = student_rollout(episode,forward_and_backward,device=device,
        auxiliary_dtype=auxiliary_dtype,detach_history=True,stop_after_last_label=False)
    if tick != len(episode):
        raise RuntimeError("recorded smoke did not forward the complete dense sequence")
    return prediction, returned_targets, backwards


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Legacy 2580-output diagnostic only. Use lane_training.smoke for CD210.")
    episode_input = parser.add_mutually_exclusive_group()
    episode_input.add_argument("--manifest", type=Path)
    episode_input.add_argument("--diagnostic-manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["native", "float32"], default="native")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    source = DEPENDENCY_ROOT / "fov/onnx_trainable.py"
    spec = importlib.util.spec_from_file_location("_lane_dataset_onnx_trainable", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ReleasedModel(args.checkpoint, precision=args.precision).to(args.device).eval()
    def parameter_digest():
        digest = hashlib.sha256()
        for name, value in model.named_parameters():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    before = parameter_digest()
    episode_report = {}
    if args.manifest or args.diagnostic_manifest:
        from .loader import DiagnosticSequence, SequenceEpisode
        loader = DiagnosticSequence if args.diagnostic_manifest else SequenceEpisode
        with loader(args.diagnostic_manifest or args.manifest) as episode:
            prediction, targets, backward_calls = recorded_rollout_backward(episode, model, device=args.device,
                auxiliary_dtype=torch.float16 if args.precision == "native" else torch.float32,
                progress=lambda tick,total: print(json.dumps(dict(forwarded_ticks=tick,total_ticks=total)),flush=True))
            episode_report = dict(manifest=str(episode.path),
                manifest_sha256=hashlib.sha256(episode.path.read_bytes()).hexdigest(),
                loader_source_sha256=hashlib.sha256(Path(__file__).with_name("loader.py").read_bytes()).hexdigest(),
                manifest_status=episode.manifest["status"], episode_file_hashes_verified=True,
                recorded_frame_count=len(episode), forwarded_tick_count=len(episode),
                label_indices=targets["label_indices"].tolist(),
                lateral_label_count=int(targets["lateral_mask"].sum()),
                supported_plan_elements=int(targets["plan_mask"].sum()),
                backward_calls=backward_calls,backward_mode="streamed_per_label_normalized_valid_tick",
                recurrent_history_detached=True,
                accepted_label_count=episode.manifest.get("accepted_label_count",len(targets["label_indices"])))
        kwargs = {name: torch.as_tensor(targets[name], device=args.device)
                  for name in ("plan_targets", "plan_mask", "lateral_targets", "lateral_mask")}
        scope = ("quarantined-recording-integration-not-corpus-qualification" if args.diagnostic_manifest
                 else "accepted-corpus-dense-student-rollout")
    else:
        state = RebelliousHopeState()
        frame = np.full((2, 6, 128, 256), 128, np.uint8)
        frame[:, :4] = 16
        inputs = state.push(frame, [1, 0, 0, 0, 0, 0, 0, 0], [1, 0], [.25, .35])
        prediction = model({key: torch.as_tensor(value, device=args.device) for key, value in inputs.items()})
        kwargs = dict(lateral_targets=torch.zeros(1, device=args.device),
                      lateral_mask=torch.ones(1, dtype=torch.bool, device=args.device))
        scope = "synthetic-contract-fixture-not-corpus-integration"
    loss, diagnostic = expert_trajectory_loss(prediction.float(), **kwargs)
    if not (args.manifest or args.diagnostic_manifest):
        loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    nonzero = sum(int(torch.count_nonzero(gradient)) for gradient in gradients)
    after = parameter_digest()
    report = dict(schema_version=1, scope=scope, checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=model.source_sha256, adapter_source_sha256=hashlib.sha256(Path(__file__).with_name("adapter.py").read_bytes()).hexdigest(),
                  model_reconstruction_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  precision=args.precision, device=args.device, torch_version=torch.__version__,
                  prediction_shape=list(prediction.shape), prediction_finite=bool(torch.isfinite(prediction).all()),
                  loss=float(loss.detach()), valid_ticks=diagnostic["valid_tick_count"],
                  gradient_tensors=len(gradients), nonzero_gradient_elements=nonzero,
                  gradients_finite=finite, parameters_unchanged=before == after,
                  parameter_sha256_before=before, parameter_sha256_after=after,
                  optimizer_steps=0, elapsed_seconds=time.perf_counter()-start,
                  note="PyTorch ONNX reconstruction gradients; native-precision arithmetic is not certified identical to device kernels.")
    report.update(episode_report)
    report["passed"] = bool(finite and nonzero > 0 and before == after and report["prediction_finite"])
    if args.diagnostic_manifest:
        report["passed"] &= episode_report["lateral_label_count"] > 0
        report["qualification_effect"] = "none; diagnostic status and masks are preserved; no publication or optimizer step"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
