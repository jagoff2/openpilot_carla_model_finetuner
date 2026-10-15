# openpilot CARLA model fine-tuner

Collect checked lane-centering examples in CARLA, fine-tune **CD210/C210M** while
new examples arrive, and export the best saved model to ONNX and native comma 3X
Tinygrad PKLs. The direct geometric objective penalizes off-center paths,
turn-cutting between prediction points, incorrect heading, and insufficient
vehicle-body clearance. Frozen original-model progress prevents a shortened path
from becoming an easy way to minimize loss.

This repository contains collection, calibration, route planning, physical QA,
label generation, data loading, differentiable CD210 loading, training, export,
tests, and local source dependencies. **CARLA itself, maps, datasets, model
weights, checkpoints, and compiled executables are not included.** Original
CD210 weights are downloaded separately and verified by SHA-256.

The default runs **both CARLA GPUs concurrently** and a separate CPU trainer
using two threads. There is no alternating collection/training schedule. The
trainer performs real AdamW updates and waits after completing its configured
passes over currently accepted data.

## Requirements

- Windows 10/11, PowerShell, Git, Python 3.12 for capture and Python 3.10 for
  training. Setup uses the Windows `py` launcher unless paths are supplied.
- Separate **CARLA 0.9.16** Windows installation and matching Python wheel, with
  Town01/02/03/04/05/06/07/10HD installed. Follow the
  [CARLA installation instructions](https://carla.readthedocs.io/en/0.9.16/start_quickstart/).
- Two GPUs. Development used two RTX 5060 Ti 16 GB cards. GPU 0 uses RPC port
  2400 and GPU 1 uses 2500; leave streaming ports 2401/2501 and traffic-manager
  ports 8400/8500 available too.
- Storage for CARLA plus data and model output. Corpus defaults are a 60 GiB
  allocation cap and 30 GiB minimum free space. Checkpoints/exports are additional.
- Internet for dependencies and CD210, or an existing verified model cache.
- Native PKL export additionally needs an online, offroad c3x with the matching
  IQPilot generation-12 split-model/Tinygrad runtime and SSH access. ONNX export
  works without a device. Linux simulator launch has not been qualified here.

## Install

Install CARLA separately, then run:

```powershell
git clone https://github.com/jagoff2/openpilot_carla_model_finetuner.git
cd openpilot_carla_model_finetuner
.\setup.ps1 -CarlaRoot 'J:\simulators\CARLA_0.9.16' -DatasetRoot 'G:\lane-centering-dataset-v1'
```

Replace those paths. Setup creates `.venv` for CARLA and `.venv-training` for CPU
training, installs this checkout, fetches CD210, and writes ignored machine
settings to `config.local.json`. It does not download CARLA or start collection.

Optional arguments accept explicit interpreters and an existing ONNX cache:

```powershell
.\setup.ps1 -CarlaRoot 'J:\simulators\CARLA_0.9.16' -DatasetRoot 'G:\lane-centering-dataset-v1' `
  -CapturePython 'C:\Python312\python.exe' -TrainingPython 'C:\Python310\python.exe' `
  -SourceExports 'D:\cached-models\cd210'
```

The cache must contain exact `driving_vision.onnx` and `driving_policy.onnx` files.
Native metadata is regenerated with **the same pinned hashes** as original CD210.
Training does not depend on downloading old QCOM executables from a private server.
`CARLA_ROOT`, `LANE_DATASET_ROOT`, and `LANE_TRAINING_PYTHON` override local settings.
Without settings, Python defaults to `data/corpus` inside the checkout.

## Prepare

```powershell
.\prepare.ps1
```

Preparation checks maps/client/model hashes, initializes storage, runs the tests
used as admission evidence, measures steering dynamics at 1.5/2.5/3.5 m/s, and
builds native route candidates and the Town06 seed route. Calibration starts or
reuses CARLA on port 2500. Preparation can take substantial time and must finish
before collection starts. Individual stages can be resumed:

```powershell
.\prepare.ps1 -Stage doctor
.\prepare.ps1 -Stage tests
.\prepare.ps1 -Stage calibrate
.\prepare.ps1 -Stage plan
```

Qualified calibration and existing plans are reused. Failed calibration is never
marked passing. Stop other simulator clients before calibration. Do not run two
checkouts against the same CARLA ports.

Configuration and qualification bind source hashes. A corpus from a different
source/config revision cannot transparently resume in a fresh checkout. This
packaged checkout does not modify the original development checkout or its jobs.

## Collect and train concurrently

```powershell
.\start.ps1
```

This opens separate visible collection and training consoles. `./start.ps1
-Prepare` performs preparation first. Alternatively run `./collect.ps1` and
`./train.ps1` in separate consoles.

The collector owns both GPUs, starts/reuses their CARLA servers, records RGB/depth,
camera history, telemetry and boundary observations, checks physical behavior and
label support, and publishes accepted episodes. Rejections are recorded and the
queue advances. Captures with recorded frames retain their payloads for recovery;
empty preflight failures retain diagnostics. After a collector correction,
completed events rejected by the corrected admission checks are retried before
unproven routes. Warmup frames and recording progress do not increase the accepted
count; publication follows label and physical checks.

The catalogue schedules routes with continuous painted boundaries through warmup
and the maneuver before routes with long unpainted gaps, distributing opportunities across scenario families and
the separate Town05 test pool. This avoids spending both workers on short,
unmarked junction routes while observable curves and straights wait. Curbs and
road edges remain eligible. Map markings affect scheduling only; RGB/depth
evidence, physical checks, split isolation and corpus quotas still decide
acceptance. `screened-pool.queue.sqlite` is a rebuildable scheduling index beside
the original catalogue, not another dataset. The console reports
`collection_candidate_selected` with the boundary priority for each new job.

The trainer follows `training-index.partial.json`, regenerates separate original
and student recurrent histories, and updates both vision and policy using accepted
**training** episodes. Four passes per episode are the default; new episodes take
priority. Validation and test data cannot enter optimizer updates. Test data is
not used to select checkpoints.

```powershell
$settings = Get-Content .\config.local.json -Raw | ConvertFrom-Json
Get-Content (Join-Path $settings.dataset_root 'reports\automatic\status.json')
Get-Content .\runs\cd210-centerline-production\status.json
```

The target is **5,000 accepted labels**: 4,000 train, 500 validation, 500 test,
with 20 labels per accepted episode and dense 20 Hz frames retained for history.
Town05 is test-only. Independent geometry, scenario families, direction, maneuver
phases and historical crop-loss coverage are release requirements. Candidate
exhaustion reports deficits; the collector does not guarantee 5,000 passing labels
or manufacture missing coverage. Final `training-index.json` appears only after
all release checks pass. Partial accepted data can train before that point.

Restart the same commands to resume. The collector reattaches exact live worker
identities. The trainer uses atomic checkpoint pointers and an OS lock. To stop
training after the current operation:

```powershell
New-Item .\runs\cd210-centerline-production\stop.request -ItemType File
```

Remove the marker before restarting. Changed training sources/configuration
require an explicit migration. Capture rejection details are in the corpus's
`reports/automatic/*.log`; trainer events are under `runs/`.

## Export on demand

```powershell
# Works while the device is offline:
.\export-c3x.ps1 -OnnxOnly

# Select current best, export, and compile on an online/offroad device:
.\export-c3x.ps1 -DeviceHost 'iq@192.168.1.95'

# Compile an already exported package:
$release = (Get-Content .\releases\cd210\latest.json -Raw | ConvertFrom-Json).directory
.\export-c3x.ps1 -Resume $release
```

Default SSH is `iq@192.168.1.95`. An offline device retains verified ONNX and returns
code 3 with resume instructions. The exporter compares retained checkpoints and
original CD210 on the same current validation cohort. **When original CD210 scores
best, it exports original weights and reports step 0.** Newest does not mean best.
Collection and training continue during export.

`releases/cd210/<timestamp>` contains the selected `checkpoint.pt`, vision/policy
ONNX, metadata, validation scores, hashes and numerical checks. Successful native
compilation adds executable `*_tinygrad.pkl` files, `bundle.json`, device reports,
and `cd210-c3x.zip`. The exporter does not activate a model or replace driving
software. Its bundle targets IQPilot's CD210 split-model interface, not a renamed
generic `supercombo.onnx`. See [export details](docs/c3x-export.md).

## Objective, data checks and limitations

The main training target is the **observed lane centerline at corresponding road
progress**, including predicted segment interiors, heading and body clearance.
Recorded speed is not a model input or the geometric target. CD210 receives camera
sequences, recurrent features, desire and traffic convention. Rebellious Hope is
not a training base or automatic fallback; legacy adapter code remains for tests.

Capture QA includes mean signed center offset <= 5 cm, p95 absolute offset <= 10 cm,
maximum absolute offset <= 20 cm, body clearance >= 15 cm, lateral acceleration
<= 3.5 m/s² and lateral jerk <= 5 m/s³. Painted lines and visible curbs/road edges
require independent RGB/depth support. These are data-admission measurements.

Improving loss and successful compilation do not establish perfect lane centering
or real-road qualification. Reports distinguish `runtime_ready` from
`driving_qualified`. No completed corpus or validated improved checkpoint is
distributed here. Synthetic camera optics are nominal assumptions.

## Code and tests

| Path | Purpose |
| --- | --- |
| `lane_dataset/` | Capture, controller, observations, labels, QA, storage and loading |
| `reports/*.py` | Operational supervisor, parallel workers and evidence processing |
| `lane_training/` | CD210 fetch/loader, objective, training and checkpoints |
| `lane_release/` | Selection, ONNX export, native compilation and verification |
| `lane_reference/` | Bundled preprocessing, archive and diagnostic source dependencies |
| `pipeline/` | Fresh-install checks, calibration and route preparation |
| `tests/` | Geometry, interfaces, admission, trainer and exporter regressions |

```powershell
.\.venv-training\Scripts\python.exe -m pytest tests -q --junitxml=reports\tests.xml
```

Tests need no live simulator; actual-model tests need the weights installed by
setup. Optional CPU Tinygrad probes take explicit runtime and Clang paths. CPU
success does not establish QCOM compatibility. Generated reports are ignored.

See the [objective](docs/direct-centerline-objective.md),
[training process](docs/production-training.md), [dataset contract](DATASET_CARD.md),
and [observability/split rules](docs/observability-and-splits.md).
The existing [MIT license](LICENSE) covers original project code. Bundled runtime
excerpts preserve their own terms in [third-party notices](THIRD_PARTY_NOTICES.md).

## Preserving the best model during unattended runs

Every better-than-stock validated candidate and every prior winner is saved in
`runs/cd210-centerline-production/preserved/checkpoints/<SHA-256>.pt`. These files
are never replaced or automatically pruned. The rotating resume slots are separate.
`best.json` changes only for a strict comparable validation improvement over stock
and the incumbent, with no regression on earlier promotion cohorts. Changed
validation cohorts trigger an incumbent comparison; ties keep the existing best.

Full-model ONNX and compiled PKL exports use separate immutable directories.
Retries cannot overwrite a successful package. `releases/cd210/best-export.json`
selects the best ONNX candidate; `best-runtime.json` retains the last successfully
compiled best package while a new candidate awaits device compilation. Final
training completion also exports the best selection, never simply the last step.
See [checkpoint retention](docs/production-training.md) and
[export selection](docs/c3x-export.md) for paths and migration instructions.

Checkpoint ranking, promotion and export selection use **lateral quality only**:
`lateral_geometry_v1 = lateral_loss + 0.25 * heading_loss + clearance_loss`.
Forward-progress error, speed changes, and other longitudinal diagnostics do not
rank candidates or break ties. `loss` and `baseline_loss` in new validation and
selection reports refer to this lateral score; `optimization_loss` separately
includes the existing training guard against shortening the plan to evade lateral
supervision. Earlier reports without `selection_metric` retain their historical
combined score and are never compared directly with new lateral scores.

The collection policy surveys road elevations and validates depth against collision
geometry at exact sensor pixels, including interior road samples visible in the
narrow camera. Collision rays start beyond the ego vehicle, and depth/road-edge
measurements use CARLA's half-pixel sensor centers. Initial depth checks allow a
bounded sensor settling interval.
Fixture rays are screened with the independently surveyed road plane so a ray
started beneath near road cannot select a buried surface. When the same-frame
semantic image confirms road, unlabelled invisible collision volumes are excluded;
named occluders and all metric depth residual checks remain enforced.
Warmup and action admission permit a declared 0.1 m/s operating tolerance around
the unchanged measured calibration range; physical lateral QA remains 3.5 m/s²
and 5 m/s³. Stopping support uses braking distance plus reaction delay.

The queue remembers repeated geometry failures and automatically retries failures
from older collection policies once. Interrupted recordings retain their image
and observation payloads. A complete maneuver prefix with its required history
and future suffix can be recovered and submitted to the normal admission checks;
incomplete maneuvers remain unaccepted. Windows directory publication retries
temporary sharing violations and never replaces an existing destination.

Run `./monitor-trainer.ps1` to watch the existing trainer's process, phase, optimizer steps, accepted split counts, and latest held-out lateral loss. It does not start another trainer. Use `-RunDirectory <path>` to monitor a different run, or `-Once` for a single status snapshot.

