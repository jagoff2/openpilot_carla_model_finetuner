# Export the best available CD210 model for c3x

**2026-10-15 quality repair:** capture policy 8, reversible data review holds, and an ego-origin-anchored centerline objective are installed. Existing trained runs cannot silently resume across this objective change. See [the investigation and restart constraints](data-quality-investigation-20261015.md).

From `your checkout directory`:

```powershell
.\export-c3x.ps1
```

This command snapshots saved checkpoints while collection and training continue,
scores them on the current accepted validation set, exports the best model, then
connects to **`iq@192.168.1.95`** to compile and verify native QCOM Tinygrad PKLs.
It uses two CPU threads locally. Both collection GPUs remain assigned to CARLA.
No model is activated and no driving software is replaced.

To build an exact preserved training step, without reranking checkpoints or
changing the best-export pointer:

```powershell
python -m lane_release.candidate --run-dir runs/cd210-centerline-production-v2 --step 71
```

Use the configured training Python environment. This produces a separate
`step71-candidate` release with verified ONNX graphs, the saved checkpoint's
validation report, and an `*-onnx-candidate.zip` source package. The ZIP contains
ONNX and metadata, not executable Tinygrad PKLs. It attempts native compilation
at the usual device address; `--onnx-only` skips that attempt. Resume native
compilation with `export-c3x.ps1 -Resume <candidate-directory>` when the device is
online. The requested step remains pinned if training later promotes another
checkpoint. Missing or ambiguous preserved steps fail explicitly.

To generate and verify ONNX without connecting to a device:

```powershell
.\export-c3x.ps1 -OnnxOnly
```

For a full-precision native candidate from an already pinned release:

```powershell
python -m lane_release.precision --candidate <candidate-directory> --output <candidate-directory>/fp32
python -m lane_release.export --resume <candidate-directory>/fp32 --device iq@192.168.1.95
```

This uses the exact FP32 checkpoint tensors, verifies the recurrent reference
against the training model, and compiles with FP32 image intermediates. It keeps
the original export intact. Verification loads the reference archive before
timing inference, so archive decompression is excluded from model latency.

The picker installer is `lane_release/install_candidate.py`. Copy it,
`selector_patch.py`, and `local_candidate.py` into the same directory on the
c3x. The package passed to it must contain `payload.json` and the successful
`compiled/device-report.json`, `compiled/bundle.json`, and four compiled
model/metadata files. A source ONNX ZIP is insufficient.

Record the checkpoint step and full SHA256 from the selected local validation
record, and inspect the device UI before installation. Obtain its SHA256 with:

```sh
sha256sum /data/iqpilot/iqpilot/ui/layouts/settings/iq_panels.py
```

Run the installer on the offroad c3x, supplying that inspected UI hash. For the
pinned step-71 checkpoint, the invocation is:

```sh
/usr/local/venv/bin/python install_candidate.py \
  --package "$COMPLETED_NATIVE_PACKAGE" \
  --expected-step 71 \
  --expected-checkpoint-sha256 29df784231f6832b894572359dd105bbf8d84d1e4bc3e7dcc25c5039ee486b3b \
  --expected-ui-sha256 "$INSPECTED_UI_SHA256"
```

Set `COMPLETED_NATIVE_PACKAGE` to the completed package directory and
`INSPECTED_UI_SHA256` to the inspected file hash. Replace the step and checkpoint
hash together when installing another candidate. The installer checks that the
payload matches both, that the device report hashes that payload, and that all
compiled artifact hashes agree. Verification requires at least 120 recurrent
frames and a passing, runtime-ready device report.

The existing **Experimental** picker lists all installed CD210 candidates. A new
entry includes its step and checkpoint hash prefix, for example
**CD210 Centerline s71 (29df7842)**. The old **CD210 Centerline s24** entry and
its files remain available. Each candidate has separate immutable model,
descriptor, payload, and report files; the installer chooses an unused model
index and refuses conflicting replacements. The new helper is installed as
`cd210_centerline_candidates_v2.py`, preserving the original step-24 helper.

Installation adds no toggle and preserves the active model selection. The UI
backup and installation receipt are saved under the package's `selector-install/`
directory. Repeating the same completed installation is idempotent. Explicit
selection in the picker rehashes the chosen candidate's artifacts before
updating the existing model Params. UI reload and selecting the new entry are
separate offroad actions; installation itself performs neither.

The installed step-71 candidate (`29df784231f6`) uses **FP32 vision and FP32
policy**. Its 120-frame native QCOM verification passed with 38.68 ms p95 and
39.06 ms maximum model-pair latency. Position XY disagreement with the exact
FP32 checkpoint reference was 0.000046 m p95 and 0.000626 m maximum. It is listed
as **CD210 Centerline s71 (29df7842)** under Experimental. The UI was reloaded
after installation; step 24 remained installed and selected. These measurements
establish export fidelity and model-pair runtime, not road-driving performance.

The previously installed step-24 candidate uses FP32 vision and FP16 policy. The helper
`mixed_device.py` combines hash-checked components and reruns 120 recurrent
frames against the exact FP32 checkpoint reference. Mixed-precision acceptance
requires whole-policy agreement (`atol=0.05`, `rtol=0.02`), lateral p95 error
at most 5 cm, lateral maximum at most 20 cm, the existing yaw limits, and pair
p95 latency at most 50 ms. Forward X differences remain reported separately.
The installed pair measured 36.9 ms p95 and 9.3 mm maximum lateral difference.

Device timing uses modeld's CPU core 7. Offroad SSH sessions inherit a
little-core mask and the big cores can be offline; the benchmark temporarily
enables core 7 and restores its previous offroad state on exit. It does not
raise CPU/GPU clock limits. These measurements cover the model pair, not the
complete driving pipeline.

When the c3x is online and offroad, compile the already exported package:

```powershell
$release = (Get-Content .\releases\cd210\latest.json -Raw | ConvertFrom-Json).directory
.\export-c3x.ps1 -Resume $release
```

Omit `-Resume` to select from the newest saved checkpoints instead. An unavailable
device leaves the local package intact, records `device-pending.json`, prints a
resume command, and exits with code 3. It does not restart training or discard an
export. `-DeviceHost` can override the default SSH address when needed.

## What “best” means

The exporter compares the committed training checkpoint, the older resume slot,
all preserved better-than-stock validation candidates, the previous best export,
and original CD210 on the same current held-out validation cohort. Training and
test examples are never used to rank candidates.

Selection uses lateral centerline loss + 0.25 * heading loss + boundary-clearance
loss (`lateral_geometry_v1`). Forward-progress drift and velocity changes are
diagnostics only and never influence ranking or tie-breaking. Every retained
candidate is rescored on this metric, including legacy exports.

A new export best must strictly improve on both stock and the current incumbent
and must not regress on any earlier promotion cohort. Ties keep the incumbent;
a stock tie keeps stock. If there is no qualifying improvement, the exporter
returns the existing best package unchanged and writes the new comparison to a
separate directory. On the first export, stock remains the fallback when no
fine-tuned candidate beats it. Existing artifacts are never overwritten by this
selection process, including when a changed validation set changes rankings.

`best-export.json` is the authoritative selected-checkpoint pointer. `latest.json`
points to the best promoted package, while `last-request.json` records the most
recent request and comparison. `best-runtime.json` retains the last successfully
compiled best package while a newer ONNX candidate awaits compilation. Previous
pointer records are retained in `history/`. `selection.json` records scores,
validation manifest hashes and checkpoint identities. These scores measure the
held-out lateral objective; partial validation does not establish driving
qualification.

## Files and verification

Each request produces an immutable checkpoint snapshot and model artifacts under
`releases\cd210\<UTC timestamp>`. This is model output, not another corpus.

- `checkpoint.pt`: selected weights and provenance, with optimizer state removed.
- `driving_vision.onnx` and `driving_policy.onnx`: native CD210 split graphs.
  Camera input remains uint8; policy inputs and original weights retain their
  FP16 export format. Shapes, output slices and 20 Hz recurrence remain CD210.
- Role-specific metadata PKLs, `export.json`, and `selection.json` bind the
  selected weights and native interface to their hashes.
- `onnx-verification.json`: exported parameters checked against the selected
  checkpoint, independent FP32 PyTorch/ONNX Runtime numerical comparisons, and
  a finite 120-frame native ONNX recurrent trace using validation footage.
- `payload.json`, `verification.npz`, and `device_build.py`: hash-bound input for
  native device compilation. Metadata PKLs alone are not executable models.
- `builds/<attempt>/compiled/`: created for each native build, containing vision and policy
  executable `*_tinygrad.pkl` files, metadata, `device-report.json`, the recorded
  device trace, and `bundle.json` when runtime checks pass.
- `builds/<attempt>/cd210-c3x.zip`: compiled package, produced only after successful device checks.
  Failed attempts retain their separate diagnostics. Resuming a successful package
  verifies its archive and returns it without rebuilding or overwriting ONNX/PKL files.

The c3x builder uses `/usr/local/venv/bin/python` and the installed Tinygrad runtime
under `/data/iqpilot/artifacts/package_runtime`. It requires device type `tizi`,
QCOM access, and offroad state. It serializes and reloads each model, verifies
captured input devices/dtypes, then runs both models with the compiled model's
own recurrent history on the held-out trace. It records exact Tinygrad source
hashes so that the compiled package's runtime dependency is identifiable.

Device verification requires finite outputs, full policy numerical agreement,
path XY p95 error at most 5 cm and maximum 20 cm, yaw p95 error at most 0.005 rad
and maximum 0.02 rad, and warm model-pair p95 latency at most 50 ms. These are
**compiled-versus-ONNX differences**, not errors relative to the road centerline.
Pair timing includes model inputs and output retrieval; it excludes camera warp
and the rest of modeld. Failure retains diagnostics and does not publish a bundle.

The bundle targets the local IQPilot generation-12 CD210 split-model interface.
Its file URIs expect the compiled models and metadata in `/data/media/0/models`
when installed through the matching model system. The exporter only builds a
package; it does not copy files into that active model directory, select a model,
or change camera processing.

`runtime_ready` means native QCOM serialization, recurrence, numerical and model
pair timing checks passed on this device. `driving_qualified` remains false:
offline export and compilation do not verify real driving, full modeld timing,
camera calibration or perfect lane centering. Native PKLs cannot be verified
while the c3x is offline. CPU compiler probes are recorded separately and never
mark a package QCOM-ready.
