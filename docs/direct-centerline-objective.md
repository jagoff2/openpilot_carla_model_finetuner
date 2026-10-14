# Direct centerline objective and CD210 default

The selected fine-tuning base is **CD210**, identified by the local bundle's
`internalName: C210M`. `lane_training.cd210.DEFAULT_CHECKPOINT` resolves to
`models/cd210/bundle/bundle.json`.
Both source ONNX files and native metadata are pinned by SHA-256. Original compiled artifact hashes are retained as provenance; training does not require those binaries.
Rebellious Hope is excluded as the training base.

The matching differentiable exports are now installed in `models/cd210`.
They are the original vision and policy ONNX files from comma's
[CD210 release](https://github.com/commaai/openpilot/commit/5b6436a90cf6902b8aaa71c2b6f3d7164d8ae391),
verified against its Git LFS SHA-256 hashes and the local bundle's checkpoint IDs,
input/output shapes and output slices. `lane_training.cd210_model.create_model`
loads both operator graphs into PyTorch with 30,003,148 trainable parameters.
It is the default factory for `lane_training.smoke`. Both GPUs remain allocated
to collection; all export verification and accepted-recording checks used CPU.

The source ONNX stores FP16 weights. Training defaults to FP32 master weights and
arithmetic; `precision='native'` retains FP16 operations with FP32 masters.
The verified numerical comparison uses independent ONNX Runtime CPU execution
of the same graph at FP32 precision. Differences from the original FP16 ONNX
are recorded separately. Compiled QCOM runtime parity and driving performance
have not been established by these checks.
The optional native FP16 path has finite outputs but does not meet the tight
FP32 comparison tolerance; use the validated default FP32 path for training.

## Objective

`lane_training.centerline.centerline_loss` accepts current-relative FRD XY paths
in metres, shaped `[labels, points, 2]`, plus optional yaw radians. The geometry
loss has no dependency on a particular checkpoint, output offset, point count,
recorded driving speed or recorded future steering command.

For each label, a **separate frozen copy of the selected original model** supplies
the distance between successive predicted positions. Accumulating those lengths
from the current observed lane station gives ordered route progress. The student
must place its predicted path at the observed lane center at those distances.
Both lateral displacement and displacement along the lane tangent are penalized.
This preserves the original model's path-length progression while teaching its
path to bend with the lane. A shorter or stationary student prediction keeps the
same target stations and support masks, so shortening cannot remove its penalty.
Original predictions with zero progress provide no spatial-path supervision.

The loss samples each predicted segment at reference-distance intervals of at
most 25 cm, including endpoints. It also penalizes heading error and the sampled
vehicle perimeter entering the 15 cm boundary margin. Where a model predicts
yaw, that yaw orients the vehicle footprint; otherwise the path direction does.
These checks penalize line hugging and chords that cut across a curve even when
the output endpoints lie on the centerline. They are sampled losses, not a
continuous geometric containment certificate or a guarantee of perfect driving.

Defaults are Smooth L1 lateral/clearance scales of 10 cm, a tangent displacement
scale of 1 m, weights of 1 for lateral/progress/clearance, and .25 for heading.
Progress and lateral penalties cannot be disabled through `LossConfig`.
The loss directly supervises XY and yaw. It does not directly supervise plan
velocity, acceleration, angular rates, uncertainty or unrelated output heads.
Their behavior must be checked when evaluating actual fine-tuned CD210 weights.

## Observable geometry and immutable recordings

`episode_corridors(SequenceEpisode(...))` verifies the accepted shard's file
hashes, then rebuilds each label's corridor from its admitted, observed boundary
knots. Each knot must have in-range camera-frame evidence at or before the label,
within its five-second input history. Unsupported intervals, future evidence,
excessive observation gaps and extrapolation are rejected or masked.

The left and right fits use only their own observed XYZ samples and sample
identities. Hidden map centerline coordinates, hidden derivatives, recorded
speed and future expert commands do not enter the geometric target. Existing
admission evidence retains its 5 cm interpolation-agreement check. The target is
transformed using the current recorded pose into the same FRD coordinates as
the model's path. Accepted files are read-only and are not rewritten.

This geometric mask is independent of the older causal expert-trajectory mask.
Already observed curves can therefore supervise farther ahead even where a
future expert command could not be certified at the current decision time.
The first accepted episode supplies 16.6 to 45.3 metres of forward observed
corridor across its 20 selected labels.

## CD210 interface

The verified metadata declares:

| Component | Inputs | Output |
| --- | --- | --- |
| Vision | `img`, `big_img`: packed uint8 `[1,12,128,256]` | `[1,1576]`; features at `1064:1576` |
| Policy | `desire_pulse [1,25,8]`, `traffic_convention [1,2]`, `features_buffer [1,25,512]` | `[1,1000]`; plan means at `0:495`, reshaped `[33,15]` |

The remaining plan values `495:990` encode uncertainty. They receive no direct
centerline loss. Plan position is columns 0:3 and yaw is column 11.

Every recorded 20 Hz tick regenerates the student's vision features. Cameras
use packed frames at `t-4,t`. The policy receives vision features from
`t-96,t-92,...,t-4,t`, including **current vision**. Rising desire pulses are
pooled in four-tick bins. Teacher and student each regenerate their own histories;
teacher features never become student inputs. There is no measured-speed,
recorded steering or `action_t` input in this CD210 interface.

`CD210SplitModel(vision, policy)` composes differentiable modules implementing
the two native components. Its training wrapper accepts 24 previous feature
entries, computes/appends current vision, and returns `[1,1512]`: native policy
output followed by 512 current features for the sequence runner. This wrapper
layout does not change the deployed CD210 model.

The default model can be loaded directly:

```python
from lane_dataset.loader import SequenceEpisode
from lane_training.cd210 import backward
from lane_training.cd210_model import create_model, save_checkpoint, export_onnx

device = 'cpu'  # Both GPUs are currently reserved for concurrent collection.
student = create_model(device=device)
original = create_model(device=device).requires_grad_(False)
with SequenceEpisode(accepted_training_manifest) as episode:
    predictions, diagnostics = backward(episode, student, original, device=device)
# Gradients are now accumulated. An explicitly configured trainer owns any
# optimizer step, checkpoint writing, split selection and validation schedule.
```

`save_checkpoint(student, 'candidate.pt', optimizer=optimizer, step=step)` saves
master weights, optimizer state and the upstream identity. `load_checkpoint`
restores them with strict state-dict matching. `export_onnx(student, directory)`
writes both original graphs with updated FP16 weights, preserving their native
interfaces and metadata. It refuses to overwrite the original source files.
The exported ONNX files need target-device compilation and deployment checks
before they can replace the installed QCOM executables.

`backward` releases each selected label's graph after accumulating its gradient,
normalizing over supported labels. Prior recurrent features are detached; the
current vision-to-policy gradient remains connected. It performs no optimizer
step. Tests compare its gradients with the corresponding complete batch loss.

## Verification commands and scope

Run the accepted-data audit without using either collection GPU:

```powershell
python -m lane_training.audit --manifest '<dataset_root>\accepted\auto-town06-road40-terminal-0\manifest.json' --report '<dataset_root>\reports\automatic\direct-centerline-cd210-interface-audit.json'
```

The audit verifies CD210 artifact hashes, accepted data integrity, geometric
target construction, center preference over shifted paths, and 1,100 dense-frame
rollouts through a small synthetic vision/policy fixture. It explicitly reports
`checkpoint_inference_performed: false` and `optimizer_steps: 0`. It tests the
data/interface/objective, not CD210 inference or driving performance.

Run actual CD210 checks with the built-in default factory:

```powershell
python -m lane_training.fetch_cd210
python -m lane_training.verify_cd210 --manifest '<dataset_root>\accepted\auto-town06-road40-terminal-0\manifest.json' --report reports/cd210-differentiable-export.json
python -m lane_training.smoke --manifest '<dataset_root>\accepted\auto-town06-road40-terminal-0\manifest.json' --report reports/cd210-accepted-backward.json
```

The commands above generate local numerical-gradient, source-ONNX parity, accepted-sequence backward, checkpoint and export reports. These reports and the recorded corpus are not distributed in this repository. A successful backward check alone does not update model weights.

The [production trainer](production-training.md) now uses this same default
factory and objective, with actual optimizer steps and resumable checkpoints
while accepted data continues to arrive.

`--model-factory path.py` remains an optional override. `--checkpoint` defaults
to the pinned CD210 bundle and `--device` defaults to CPU. No fallback to
Rebellious Hope exists in this training path. A regression test takes one
optimizer step on a disposable model copy and verifies lower centerline loss,
save/reload and export without changing the selected source checkpoint.

Existing accepted manifests retain their original source/configuration bindings. A different checkout or configuration requires an explicit migration, not a silent restart against old data.
