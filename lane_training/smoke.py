"""Actual accepted-episode centerline forward/backward, with no weight updates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

from lane_dataset.loader import SequenceEpisode
from .cd210 import DEFAULT_CHECKPOINT, backward, verify_checkpoint


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="CD210 bundle.json; all four artifacts are hash-verified")
    parser.add_argument("--model-factory", type=Path, default=Path(__file__).with_name('cd210_model.py'),
        help="Python module exporting create_model(bundle_path, device, precision), returning CD210SplitModel with actual checkpoint weights")
    parser.add_argument("--device", default="cpu", help="both collection GPUs remain reserved for capture")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--precision", choices=["native", "float32"], default="float32")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    start = time.perf_counter()
    checkpoint = verify_checkpoint(args.checkpoint)
    source = args.model_factory.resolve()
    spec = importlib.util.spec_from_file_location("_cd210_trainable_factory", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    student = module.create_model(args.checkpoint, device=args.device, precision=args.precision).eval()
    reference = module.create_model(args.checkpoint, device=args.device, precision=args.precision).eval()
    reference.requires_grad_(False)
    before, reference_before = parameter_digest(student), parameter_digest(reference)
    with SequenceEpisode(args.manifest) as episode:
        prediction, diagnostic = backward(episode, student, reference, device=args.device,
            auxiliary_dtype=torch.float16 if args.precision == "native" else torch.float32,
            complete_sequence=True,
            progress=lambda stage, tick, total: print(json.dumps(dict(stage=stage, ticks=tick, total=total)), flush=True))
        manifest_hash = hashlib.sha256(episode.path.read_bytes()).hexdigest()
        frame_count = len(episode)
    gradients = [p.grad for p in student.parameters() if p.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
    nonzero = sum(int(torch.count_nonzero(g)) for g in gradients)
    after, reference_after = parameter_digest(student), parameter_digest(reference)
    report = dict(schema_version=1, scope="accepted-corpus-direct-centerline-objective-backward",
        checkpoint=str(args.checkpoint.resolve()), checkpoint_binding=checkpoint,
        differentiable_export_binding=getattr(student, 'checkpoint_binding', None),
        manifest=str(args.manifest.resolve()), manifest_sha256=manifest_hash,
        episode_file_hashes_verified=True, recorded_frame_count=frame_count,
        device=args.device, precision=args.precision, torch_version=torch.__version__,
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        model_reconstruction_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        prediction_finite=bool(torch.isfinite(prediction).all()), gradient_tensors=len(gradients),
        nonzero_gradient_elements=nonzero, gradients_finite=finite,
        gradients_by_module={role: dict(tensors=sum(p.grad is not None for p in getattr(student,role).parameters()),
            nonzero_elements=sum(int(torch.count_nonzero(p.grad)) for p in getattr(student,role).parameters() if p.grad is not None))
            for role in ('vision','policy')},
        parameter_sha256_before=before, parameter_sha256_after=after,
        reference_sha256_before=reference_before, reference_sha256_after=reference_after,
        parameters_unchanged=before == after, reference_parameters_unchanged=reference_before == reference_after,
        optimizer_steps=0, elapsed_seconds=time.perf_counter()-start, **diagnostic)
    report["passed"] = bool(finite and nonzero > 0 and before == after and reference_before == reference_after
        and not diagnostic["reference_has_gradients"] and report["prediction_finite"]
        and diagnostic["reference_ticks"] == frame_count and diagnostic["student_ticks"] == frame_count
        and diagnostic["backward_calls"] == diagnostic["valid_labels"])
    report["qualification_effect"] = "objective integration only; no optimizer steps, corpus qualification or driving-performance claim"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "label_diagnostics"}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
