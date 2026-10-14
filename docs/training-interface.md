# Training interface

The selected training base is **CD210/C210M**. Its direct centerline objective, verified source ONNX exports, trainable PyTorch loader and vision/policy recurrence are documented in [Direct centerline objective](direct-centerline-objective.md). Accepted episodes are listed in `<dataset_root>\training-index.partial.json`. The matching exports are installed in `models/cd210`; `lane_training.cd210_model.create_model()` loads the actual checkpoint for fine-tuning.

The remainder of this page documents the historical 2,580-output development adapter and trajectory-imitation tests. That layout and its Rebellious Hope checkpoint are not the selected CD210 training configuration. Canonical recordings and geometric targets remain usable through the CD210 adapter.

## Packed cameras and model inputs

`StockPreprocessor.pack_rgb(narrow_rgb, wide_rgb)` accepts uint8 RGB images and returns uint8 `[2,6,128,256]`, with narrow first. Synthetic RGB conversion uses the exact pinned `rgb_to_nv12_bt601_limited` function from the reference camera probe: float64 BT.601 limited-range conversion, round-half-even luma, and 2x2 chroma averaging before rounding. The resulting NV12 is warped and packed using the existing stock MED/SBIG implementation. There is no widened field of view or additional crop.

Each packed frame has four luma parity planes followed by U and V. Images remain uint8 at the model boundary; the native model performs its normalization. The synthetic camera intrinsics are the local nominal assumptions: 1928x1208 sources, focal lengths 2648 narrow and 567 wide. Projection/crop arithmetic has a reference parity check; these values are not a physical calibration to a real comma camera.

`student_rollout` calls the student at every recorded 20 Hz tick with exactly these keys:

| Key | Shape | Content |
| --- | --- | --- |
| `img` | `[1,12,128,256]` uint8 | Narrow packed frames at `t-4,t`, old frame first |
| `big_img` | `[1,12,128,256]` uint8 | Wide packed frames at `t-4,t`, old frame first |
| `desire_pulse` | `[1,25,8]` floating | Rising edges of recorded desire, pooled over four-tick bins |
| `traffic_convention` | `[1,2]` floating | Recorded one-hot traffic convention |
| `action_t` | `[1,2]` floating | Recorded lateral and longitudinal action horizons in seconds |
| `features_buffer` | `[1,24,512]` floating | Student-generated features at `t-96,t-92,...,t-4` |

The callback must return a Torch tensor `[1,2580]`. Auxiliary dtype defaults to float32. Pass `auxiliary_dtype=torch.float16` for the native half-precision checkpoint reconstruction. Image inputs remain uint8 in either case.

Episode histories start at zero. At least five seconds precede the first accepted label. Dense ticks regenerate recurrent features from the current student output slice `2066:2578`; no teacher features are stored in or accepted from the dataset. The approximately 4.8-second feature history describes an input buffer, not proof that the model remembers a particular cue.

The loader converts held route desires to rising-edge pulses using the deployed temporal assembly. Turn intent must precede decisions that need it. Admission uses explicit reviewed intent-required indices/ranges for intersection decisions; a held desire is not treated as a fresh pulse at every tick.

## Accepted shard schema

```python
SequenceEpisode(manifest_path, *, require_retention=True, verify_hashes=True)
```

`manifest_path` may be an episode directory or its `manifest.json`. Only `schema_version: 1` and `status: "accepted"` are loadable. The manifest declares `frames_file`, `labels_file`, `archives.narrow`, `archives.wide`, and a `files` mapping from relative paths to SHA-256 digests. Both archive entries contain `path` and `sha256` and must agree with `files`.

By default every declared file is hashed before parsing frame metadata or loading targets. The inventory must bind both camera archives, frame metadata, and labels; paths must remain inside the shard. `verify_hashes=False` skips byte hashing but still requires that complete binding inventory. Production callers should retain the default.

Camera archives use the existing bounded-memory `FrameArchiveReader` format and hold lossless uint8 `[6,128,256]` frames. Metadata is JSONL with one row per frame:

```json
{"index":0,"frame_id":1234,"timestamp":61.7,"desire":[1,0,0,0,0,0,0,0],"traffic_convention":[1,0],"action_t":[0.15,0.2]}
```

The example horizons are illustrative; training uses actual recorded values. Frame IDs and indices must be consecutive, timestamps must advance by 0.05 seconds, and desire/traffic vectors must be one-hot. Missing, duplicate, or unordered frames fail loading.

`labels.npz` contains:

| Array | Shape and meaning |
| --- | --- |
| `label_indices` | `[L]` ordered unique integer frame indices on the 2 Hz grid (`index % 10 == 0`) |
| `plan_targets` | `[L,33,15]` floating canonical physical trajectory |
| `plan_mask` | `[L,33,15]` boolean per-element support |
| `current_speed` | `[L]` finite nonnegative m/s at the decision tick |
| `commanded_curvature` | `[L]` finite expert command in 1/m at the recorded lateral horizon |
| `lateral_mask` | `[L]` boolean action support |
| `lateral_targets` | Optional `[L]`; if present, checked against the canonical encoding |
| `longitudinal_targets`, `longitudinal_mask` | Optional paired `[L]` arrays; omitted for this corpus unless independently qualified |

Each accepted row must contain supported supervision. Fully unsupported rows belong in diagnostics. By default the first label requires at least five seconds of preceding frames and the last label requires at least ten seconds of subsequent frames. The archive reader also validates frame counts, dtype, and packed shape.

`episode.frame(i)` returns `[2,6,128,256]`. Iteration returns `(packed_frame, metadata)`. `episode.selected_targets()` returns selected plan/action targets and masks, label indices, and recorded `action_t`; it does not expose simulator geometry as student input. `episode.validate_all_frames()` checks the complete archives. Use the loader as a context manager to close both archives.

`DiagnosticSequence` is a separate explicit inspection class with the same constructor signature. It requires `status: "development"` and `accepted_label_count: 0`. It preserves all hash, frame, archive, target, and retention checks but permits unsupported diagnostic rows without altering their masks. It cannot load an accepted shard; the default `SequenceEpisode` cannot load a diagnostic shard. A diagnostic read does not change any artifact or acceptance status.

## Physical plans and independent action targets

Plan timestamps are `10 * (arange(33) / 32)**2` seconds. Columns are:

| Columns | Quantity | Unit/frame |
| --- | --- | --- |
| `0:3` | Position | m, relative to the current vehicle, forward/right/down |
| `3:6` | Velocity | m/s, current forward/right/down frame |
| `6:9` | Acceleration | m/s², current forward/right/down frame |
| `9:12` | Roll, pitch, yaw | rad, current-relative orientation with angular unwrapping |
| `12:15` | Angular velocity | rad/s, current forward/right/down frame |

`dense_to_plan(timestamps, position_world, velocity_world, rotation_world, label_indices, observable_support)` takes recorded CARLA world positions/velocities and CARLA roll/pitch/yaw in radians. One Hermite interpolant supplies coherent position, velocity, and acceleration; SLERP supplies relative orientation and interval angular velocity. The initial relative position and orientation are zero. Unavailable future times, missing-tick intervals, and unsupported elements are masked. Time-parametric plans can represent hairpins without a single-valued forward-distance assumption.

Admission separately establishes observational support using rendered, depth-checked lane evidence in actual stock-packed images. Paired visible boundary knots may support bounded interpolation across marking gaps; these inferred intervals are explicitly distinct from directly observed paint, curb, or road-edge pixels. The approved edge path requires RGB contrast, adjacent road/exterior semantic classes, and depth agreement on both sides. Semantic masks and depth are capture QA only; model inputs remain stock RGB. Admission replays edge contrast from retained stock-packed images and bound semantic patch proofs. Hidden map geometry validates the reconstructed corridor but does not supply controller command geometry. Future support is causal to the decision and can end before ten seconds.

The controller and admission share `geometry.MAX_OBSERVED_KNOT_GAP_M` (8 m), accommodating measured Town04 broken-marking gaps of roughly 6.5 to 7.5 m. Reconstructed boundaries and center must still agree with independent full-map validation within 5 cm. Increasing the supported paint-gap model does not change that quality gate. Captured interpolation provenance must match the current bound, so older 5 m captures cannot silently acquire a longer controller preview.

Geometric support alone is insufficient for a recorded future trajectory: a later controller command can incorporate a newly visible cue. Admission additionally checks the hash-bound `command-provenance.jsonl.gz` ledger, one record per recorded index, with `index`, `frame_id`, `timestamp`, `command_accepted`, `source_indices`, and `evidence_frames`. A command applies after its recorded physical snapshot. Every intervening command must use only paired knots already supported at the original decision, and its evidence frames must be causal and time-aligned. The first failing transition ends supported future states. Hermite/SLERP interpolation additionally requires its right endpoint; acceleration and angular rates at an exact tick require the next endpoint too.

Older recordings without the ledger mask every future component and the interpolation-derived initial acceleration/rates. Initial position, velocity, and orientation may remain supervised if geometrically supported. Immediate lateral action has its own independent evidence and mask. `label_records` retain per-element plan masks and the causality restriction reason so unobserved future control influence cannot become a hidden target assumption.

For raw output **2062**, the target is:

```python
target = max(1.0, current_speed) ** 2 * expert_commanded_curvature
```

Positive curvature follows positive yaw in forward/right/down coordinates. This uses the speed at the decision tick and the expert command for the recorded action horizon. Steering fraction, measured lateral acceleration, and future speed are not interchangeable target values.

The actual runtime decoder divides raw 2062 by `max(1, current_speed)**2` only above its low-speed boundary. At `current_speed <= 0.3 m/s`, it holds the previous desired curvature. Admission therefore masks new lateral-action supervision at or below 0.3 m/s. It also requires calibrated steering/speed range, a correctly aligned horizon, an accepted command, and causal support for the command's geometry.

The adapter supervises plan means at raw `917:1412`, lateral mean at `2062`, and optional independently qualified longitudinal mean at `2063`. Every unrelated head, uncertainty output, and unqualified action component receives no direct loss from this corpus. Recurrent outputs can still receive gradients through their later influence when full recurrent backpropagation is selected.

## Rollout and loss API

```python
student_rollout(episode, student, *, device="cpu", auxiliary_dtype=None,
                detach_history=True, stop_after_last_label=True)
```

This returns `(predictions[L,2580], selected_targets)`. All dense ticks through the final label run through the student. The retained future suffix constructs/audits labels; it is not inferred by default. `stop_after_last_label=False` also forwards that suffix.

`detach_history=True` regenerates the correct recurrent values while truncating their autograd connections; unlabelled ticks run without gradients. `False` retains the recurrent graph and costs substantially more memory. Neither mode creates an optimizer or updates weights.

The loss accepts one or more independently masked heads:

```python
expert_trajectory_loss(
    predictions, plan_targets=None, plan_mask=None,
    lateral_targets=None, lateral_mask=None,
    longitudinal_targets=None, longitudinal_mask=None,
    candidate_mask=None, valid_tick_mask=None,
    plan_scales=None, lateral_scale=2.5, longitudinal_scale=2.0,
)
```

Targets/masks must be Torch tensors on the prediction device; masks are boolean. Plan shapes are `[L,K,33,15]`, action shapes `[L,K]`; the singleton candidate dimension may be omitted. Available heads receive equal weight within each candidate, then one candidate is selected jointly across heads. There is no requirement to invent longitudinal labels when only lateral supervision exists. Unsupported elements produce zero loss and gradient, including masked NaNs; an entirely masked batch produces a differentiable zero. The return value is `(loss, diagnostics)`.

A complete caller helper, with a compatible student supplied by the caller:

```python
import torch
from lane_dataset.adapter import expert_trajectory_loss
from lane_dataset.loader import SequenceEpisode, student_rollout

def forward_backward_check(manifest_path, student, device="cuda:0"):
    student.zero_grad(set_to_none=True)
    with SequenceEpisode(manifest_path) as episode:
        predictions, targets = student_rollout(
            episode, student, device=device, auxiliary_dtype=torch.float16,
            detach_history=True,
        )
    supervised = {
        key: torch.as_tensor(targets[key], device=device)
        for key in ("plan_targets", "plan_mask", "lateral_targets", "lateral_mask")
    }
    loss, diagnostics = expert_trajectory_loss(predictions, **supervised)
    loss.backward()
    return float(loss.detach()), diagnostics
```

This helper requires an accepted shard and performs no optimizer step. The standalone smoke command also hashes all parameters before and after backward:

```powershell
python -m lane_dataset.smoke --device cuda:0 --report reports/checkpoint-contract-smoke-current.json
# Explicit integration check on a real, unaccepted recording with valid lateral labels:
python -m lane_dataset.smoke --diagnostic-manifest '<development-manifest-path>' --device cuda:0 --report reports/diagnostic-episode-smoke.json
# Once a real accepted manifest exists:
python -m lane_dataset.smoke --manifest '<accepted-manifest-path>' --device cuda:0 --report reports/accepted-episode-smoke.json
```

The [completed smoke report](../reports/checkpoint-contract-smoke.json) used the real pinned checkpoint and a synthetic contract fixture. It recorded finite gradients and unchanged parameters. It does not certify deployed GPU kernel parity, accepted episode recurrence, complete corpus loading, or improved driving performance. Those require their own measurements.

The manifest flags are mutually exclusive. Both recorded modes forward the entire dense sequence, including the suffix. Diagnostic reports use scope `quarantined-recording-integration-not-corpus-qualification`, preserve zero accepted labels, and require actual supported lateral supervision to pass. They record verified artifact hashes, processed frame count, supported mask counts, finite gradients, and unchanged model parameters. They do not relax admission or publish a training corpus.

The recorded smoke path uses `recorded_rollout_backward`: it immediately backpropagates each valid selected row, divided by the total valid-row count, and releases that graph before continuing. This matches the detached-history batch loss while limiting retained activation memory to one labelled forward pass. A regression compares its gradients and predictions against ordinary batch loss and checks that the entire retained suffix is forwarded.

The [completed real recording check](../reports/checkpoint-recording-smoke.json) forwarded all 600 frames through the pinned checkpoint on CUDA0 in 23.26 seconds. It retained the development manifest and its original masks: 20 diagnostic rows, seven valid lateral/valid loss ticks, and 615 supported plan elements. Seven backward calls produced 355 finite gradient tensors with 27,579,537 nonzero elements. The before/after parameter SHA-256 values match and the optimizer-step count is zero. This is measured integration evidence for one quarantined recording, with zero accepted labels.
