# Dataset contract

This repository generates synthetic lane-centering and turn-cut-prevention data.
It distributes no recorded corpus and makes no completed-release claim.

- Target: 5,000 accepted labels, split 4,000 training / 500 validation / 500 test.
- Twenty labels per episode, with dense 20 Hz stock MED/SBIG camera history,
  telemetry, observed boundary geometry, timestamps and independent evidence.
- Families: straight recovery, constant-radius bends, tightening S bends, severe
  hairpins and intersection turns, with approach/entry/apex/exit/recovery phases.
- Town05 is test-only. Town01/02/03/04/06/07/10HD supply train/validation routes.
  Overlapping roads cannot count as independent geometry across splits.
- At least 25% measured historical crop-loss support per split is required.
  Unsupported geometry is masked; missing evidence is never invented.
- Physical QA: mean signed offset <= 0.05 m, p95 absolute offset <= 0.10 m,
  maximum absolute offset <= 0.20 m, body clearance >= 0.15 m, lateral
  acceleration <= 3.5 m/s? and raw lateral jerk <= 5 m/s?.
- Painted boundaries and visible curbs/road edges require RGB/depth evidence.
  Calibration, depth fixtures, dynamics, complete phases and label support
  are checked before admission. Failed captures remain unaccepted.

The selected model is CD210/C210M. Its primary objective directly supervises
observed centerline geometry, heading, segment interiors and body clearance,
anchored to the frozen original model's progress. Canonical physical trajectories
and legacy action labels remain diagnostics, not the primary CD210 objective.

`training-index.partial.json` lists independently accepted episodes during
collection. `training-index.json` is published after all release checks pass.
Accepted manifests bind payloads, source/configuration, calibration, geometry,
split and qualification evidence. Accepted shards are immutable. Automated
measured review is identified explicitly and does not claim human review.

Camera optics/rendering are nominal synthetic assumptions. Simulation, partial
validation, gradients and compilation do not establish real-device calibration,
road-domain generalization or perfect driving. Runtime and driving qualification
remain separate. All data/model output paths are selected locally through setup.

See [README](README.md), [objective](docs/direct-centerline-objective.md), and
[observability/splits](docs/observability-and-splits.md).
