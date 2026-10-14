# Export the best available CD210 model for c3x

From `your checkout directory`:

```powershell
.\export-c3x.ps1
```

This command snapshots saved checkpoints while collection and training continue,
scores them on the current accepted validation set, exports the best model, then
connects to **`iq@192.168.1.95`** to compile and verify native QCOM Tinygrad PKLs.
It uses two CPU threads locally. Both collection GPUs remain assigned to CARLA.
No model is activated and no driving software is replaced.

The device is currently offline. Generate and verify ONNX without connecting:

```powershell
.\export-c3x.ps1 -OnnxOnly
```

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

The exporter compares the committed training checkpoint, an older retained slot
when available, the trainer's promoted `best.pt`, the previous best export, and
original CD210. Every candidate is evaluated against the **same current held-out
validation cohort**. No training or test examples are used to rank candidates.
It selects the lowest direct centerline loss, with newer steps breaking exact
ties. It cannot recover checkpoints already discarded by the trainer.

If no fine-tuned candidate improves on original CD210, the export contains
original CD210 weights and explicitly reports `checkpoint_kind: original_cd210`
and step 0. `selection.json` records candidate scores, the frozen baseline score,
validation manifest hashes and checkpoint identities. A partial validation set
does not establish general driving improvement.

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
- `compiled\`: created after native compilation, containing vision and policy
  executable `*_tinygrad.pkl` files, metadata, `device-report.json`, the recorded
  device trace, and `bundle.json` when runtime checks pass.
- `cd210-c3x.zip`: compiled package, produced only after successful device checks.

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
