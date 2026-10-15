# Unmarked-road lateral behavior

The intended behavior is right-hand travel on a two-way road with no painted
lane divider: follow the center of the right half, leave that nominal trajectory
only for an observed obstruction, then return to it after clearing the obstacle.
The full road midpoint is not the driving target.

For corresponding observed outer road edges L and R, the nominal target is
`0.25 L + 0.75 R`. The inferred divider is `(L + R) / 2`. The divider is derived
geometry, never falsely recorded as visible paint. Coordinates and edge ordering
follow the direction of travel, including reverse traversals and curves.

## Implemented extension

`lane_behavior` is a separate package so development does not invalidate the
active collector's captured source hashes or the running v2 trainer's source
binding.

- `scenarios.py`: unmarked OpenDRIVE straights and curves, explicit physical
  edges, both directions, clear road, parked car, and standing pedestrian cases.
  Geometry variants and appearances retain the same split group. Candidate
  plans remain development-only until rendered and physical qualification.
- `target.py`: deterministic nominal path, smooth obstacle departure and
  recovery, actual ego footprint, road containment, and swept obstacle checks.
  These are static-scene targets with causal evidence. Unsupported/moving
  hazards, insufficient observation, or inadequate clearance mask moving labels.
- `evidence.py`: raw RGB/depth/instance identities, exact sensor frame alignment,
  stock packed-crop availability, and measured stationary history. Visible
  surface evidence is distinct from the disclosed conservative simulator
  footprint used for QA. The unseen side of an object is not claimed observed.
- `records.py`: immutable geometry record, named external proof hashes,
  deterministic replay, and world-to-current-ego target reconstruction.
- `grounding.py`: place the entire parked vehicle on the road, at least 0.15 m
  from the surveyed curb, then settle it under physics, brake and handbrake.
  Retained wheel support, collision rays and stationary telemetry are replayed.
- `passing.py` and `free_space.py`: source-bound static parked-car passing
  permission. Cropped RGB/depth/semantic evidence must establish the clear road
  surface; a separate complete controlled-scene inventory verifies the fixture.
  Hidden opposing traffic is never inferred absent from inventory alone.
- `loss.py`: differentiable right-side placement, avoidance, recovery, full-body
  obstacle separation and road-boundary penalties. The first plan point remains
  at the ego origin. Fixed original-model progress prevents a shorter student
  plan from hiding lateral mistakes. No speedometer input is introduced.
- `window.py` and `future_loss.py`: supervise only the currently observed future
  interval when the model's camera crop cannot see road edges beside the ego.
  A frozen original prediction fixes its forward crossing of the observed
  target, then metric path distance fixes correspondence. The unseen prefix is
  not extrapolated. Origin anchoring and supported progress remain enforced.
- `curb_surface.py`, `curb_evidence.py`, and `surveyed_plan.py`: independently
  survey native curb collision geometry, reconstruct cropped RGB/depth curb
  observations, and preserve source maps, measured geometry, and artifact hashes.
  Physical survey interpolation is QA geometry, never a visibility claim.
- `capture.py`, `teacher.py`, and `teacher_quality.py`: record actual vehicle
  physics following an explicitly privileged simulation teacher. Full-body
  road containment, obstacle clearance, lateral acceleration/jerk, all avoidance
  phases and five metres of nominal recovery are checked independently.
- `reprocess.py`, `admission.py`, and `loader.py`: replay saved observations and
  physical evidence before publishing separate behavior manifests. A geometry
  record alone never grants acceptance or changes legacy training labels.
- `world_access.py`: reserve one simulator port after its current worker drains.
  The other GPU continues collection. Expiring reservations bind the exact
  owning process and supplement the simulator's exclusive world lock.
- `split_guard.py`: serialize publication with the ordinary collector and check
  whole native roads plus connected junctions. Another appearance or obstacle
  on the same road does not become independent validation data.
- `mixed.py` and `mixed_runner.py`: combine independently admitted ordinary-lane
  and unmarked-road sequences with explicit objective weights. A checkpoint
  must pass separate held-out comparisons for both kinds before promotion.
- `collect.py`: immutable, fixed-split native capture/admission queue with port
  reservations, disk headroom checks and append-only attempt results.
- `mixed_follow.py`: wait for admitted train/validation cohorts, update CD210 as
  data arrives, and resume the same isolated model/optimizer checkpoint chain.
- `export.py`: select a validated mixed best using its own immutable receipts,
  then prepare the existing CD210 ONNX and device-build package. Mixed training
  results never overwrite an earlier production run or its export.
- `control.py`: the original observed-target MPC adapter for right-half-only
  targets. Native passing capture uses the permission-aware physical teacher.
- `quality.py`: actual-telemetry tracking and swept-body checks, complete event
  coverage, and at least five metres of correctly offset post-avoidance travel.
- `qualification_probe.py`: bounded stationary rendering probe with exclusive
  world ownership, aligned raw/packed payloads, road-edge measurements and
  replayable obstacle evidence. It produces no training labels.

Default margins are 0.20 m from the permitted road boundaries, 0.60 m from a
parked vehicle, and 1.20 m from a standing pedestrian, measured from the vehicle
body. These are explicit simulation settings, not general road-driving safety
guarantees. Existing physical limits remain 3.5 m/s² lateral acceleration and
5 m/s³ lateral jerk.

The default contract keeps the entire ego body in the right half. The grounded
parked-car queue explicitly allows up to 1.0 m of opposing road width only during
the designated departure, pass and return, and only within the observed clear
surface intervals. The whole body must retain road and obstacle clearance.
Nominal driving and post-avoidance recovery still use `0.25 L + 0.75 R`; this
permission does not move the default driving target to the whole road center.
Missing clear-surface evidence or insufficient width yields no passing label.

Parked vehicles must be fully on the road beside the curb, not on the sidewalk
or suspended over its edge. The scene currently contains only the ego and its
designated static fixture. Pedestrian cases support stationary, physically
grounded roadside people and retain the right-half restriction. Moving/crossing
pedestrians and oncoming traffic require additional temporal prediction and a
separate yield policy; a historically stationary person is not assumed unable
to move.

Native pavement support is checked against the surveyed physical road. A
collision surface tagged `Ground` can qualify only inside that road ribbon and
within 0.04 m of its surveyed elevation. The first physical hit must qualify;
an occluder cannot be skipped to use road beneath it. Ground rays temporarily
exclude only the parked fixture itself, restore its collisions before any world
tick, and retain unchanged before/after pose and frame evidence. Settling and
recording keep physical collisions enabled.

The physical teacher uses an extra 0.10 m planning reserve around obstacles;
this does not change admission's actual clearance requirement. Steering angle,
rate and acceleration are bounded, including the acceleration contribution of
the calibrated vehicle reference point ahead of its rear axle. Actual measured
lateral acceleration and jerk are checked again after the entire recording.
Recovery must contain a continuous five-metre correctly offset segment. Its
start may settle within two metres after the planned return, while the same
heading, offset, clearance and whole-recording physical limits still apply.

## Generate development scenarios

Run from the project directory with the CARLA Python environment:

```powershell
.\.venv\Scripts\python.exe -m lane_behavior.scenarios `
  --output 'G:\lane-centering-dataset-v1\reports\unmarked' `
  --shape straight --lane-width 4.5 --split development
```

Shapes also include `constant_radius`, `s_bend`, `tightening`, and `hairpin`.
`--lane-id 1` reverses traversal. All tasks explicitly declare that existing v8
capture cannot consume them unchanged. Never append these candidate tasks to
the ordinary automatic queue and count them as supported collection.

Native development candidates are also available for Town04 road 51 in both
directions. The generated OpenDRIVE fixture failed rendered road/depth checks
on this installation; it is not a qualified collection backend. Native Town04
provided valid depth and parked-actor evidence. The original XML shoulder edge
and an inner flat material seam both differed from the physical curb toe. The
correct native curb model includes its steep face and top bevel. Independent
collision replay passed 210 of 218 cross-sections; pole-occluded samples remain
excluded. The sampled rendered toe and collision toe agreed within 1.3 mm.

The stock camera crop initially exposes a paired curb span about 18 metres
ahead. That evidence can supervise future right-side placement without claiming
that the current ego footprint lies inside visible road geometry. The cropped
detector retains its RGB contrast threshold and a combined five-centimetre toe
error budget. Moving capture still has to establish adequate causal support.

Use the probe only on a released CARLA world, with explicit `--port`, `--root`,
`--plan`, and `--case`. A configured collection port requires the reservation
protocol (or a verified paused supervisor), a drained worker, an empty world,
and the same OS world lock used by normal capture. Release reservations in a
`finally` block. Frozen actors in a stationary probe are only a rendering
fixture, never a physics-driven expert demonstration.

`reports.capture_unmarked_reserved` reserves a port and runs one bounded native
physical case with `--root`, `--plan`, `--port`, `--case`, `--start-s`, and
`--ticks`. The development capture retains raw observations, exact packed model
inputs, actual telemetry, external survey sources and executed source hashes.
`python -m lane_behavior.reprocess CAPTURE` performs observation replay;
`python -m lane_behavior.admission CAPTURE --root CORPUS_ROOT` audits admission.
Publishing additionally requires an explicit immutable geometry split assignment
and `--publish`. This is a development qualification path, not an ordinary v8
automatic-queue task.

## Native collection queue

The grounded v2 qualification queue contains parked-car and static-pedestrian
cases on Town04 road 51 for training and independent road 52 for validation.
Town05 stays test-only. Each case records 900 ticks at 20 Hz. Parked cases budget
up to 14 m for each transition, five metres of nominal recovery, five seconds
of history and ten seconds of recording suffix. The actual actor dimensions,
settled placement and teacher path must pass the budget check before recording.

From the checkout, validate the existing local queue without starting a process:

```powershell
$datasetRoot = 'G:\lane-centering-dataset-v1'
$queue = Join-Path $datasetRoot 'reports\unmarked\queues\native-grounded-v2-20261015T141939Z.json'
$capturePython = '.\.venv\Scripts\python.exe'
$trainingPython = (Resolve-Path '.\.venv-training\Scripts\python.exe').Path
& $capturePython -m lane_behavior.collect $queue --root $datasetRoot
```

Execute that immutable queue using one reserved simulator port:

```powershell
& $capturePython -u -m lane_behavior.collect $queue --root $datasetRoot `
  --port 2500 --loader-python $trainingPython --run
```

The reservation waits for the existing job to drain, holds the same world lock,
and is released before CPU admission. Collection on the other GPU continues.
The queue keeps attempt results under `reports/unmarked/automatic/attempts` and
does not silently rerun an incomplete or rejected attempt. Repeating the command
skips existing job/plan/source identities. Create
`reports/unmarked/automatic/STOP` to stop between jobs. Disk checks reserve the
raw recording, publication copy, 10 GiB headroom and the legacy collector's
reserve. Plans, surveys and corpus-local queue artifacts are not distributed
with the source package; another installation needs its own bound qualification
artifacts and explicit train/validation assignments.

## Training interface

`records.training_corridor(record, verified_evidence_bindings=...)` rebuilds a
`BehaviorCorridor`. It requires immutable external artifact bindings but does
not independently grant episode admission. Feed qualified corridors and the
chosen model's explicit plan adapter to `loss.model_behavior_loss`.

The behavior objective has its own version and must be bound into checkpoint
metadata. Preserve prior checkpoints and compare candidates on a fixed held-out
cohort containing both ordinary lane geometry and these new behavior cases.
Do not compare a new behavior score directly with the old v2 selection metric.

The default mixed recipe gives the ordinary-lane and unmarked-road objectives
equal weight, independent of their relative label counts. It uses CD210's exact
camera and recurrent-history interface; teacher steering, simulation geometry
and measured speed are not model inputs. Gradients update the chosen model's
vision and policy parameters. Candidate checkpoints and validation receipts are
append-only, and a worse validation result cannot replace the best pointer.

Read-only qualification, followed by an explicitly bounded new training run:

```powershell
python -m lane_behavior.mixed_runner --corpus 'G:\lane-centering-dataset-v1'
python -m lane_behavior.mixed_runner --corpus 'G:\lane-centering-dataset-v1' `
  --run-dir 'runs\cd210-mixed-example' --steps 4
```

Use the Torch-enabled interpreter. Both train and validation must contain
independently admitted examples of both kinds. An existing checkpoint may seed
a new run only with its exact SHA-256 and traceable training lineage. Test data
is excluded from optimizer steps and checkpoint selection.

To follow accepted data unattended in a separate console:

```powershell
$datasetRoot = 'G:\lane-centering-dataset-v1'
$trainingPython = (Resolve-Path '.\.venv-training\Scripts\python.exe').Path
& $trainingPython -u -m lane_behavior.mixed_follow --corpus $datasetRoot `
  --run-dir 'runs\cd210-unmarked-mixed' --cpu-threads 2
```

This waits without loading a model until both objectives have independently
admitted training and validation cohorts. It then performs real vision/policy
optimizer updates on CPU while collection continues. Reuse the same command and
run directory to resume weights, AdamW state and replay counts. Create
`runs/cd210-unmarked-mixed/stop.request` to stop at a saved checkpoint; remove
that request before resuming. `status.json` and `events.jsonl` in the run directory
report waiting, updates, validation and promotion. A separate run directory
preserves the existing ordinary-lane trainer and its best candidates. Source or
objective changes require a new bound run rather than silently resuming it.

Installed console commands include `lane-unmarked-collect` and
`lane-unmarked-follow`, equivalent to the module commands above.

Prepare a local export from a mixed run's validated best:

```powershell
python -m lane_behavior.export --run-dir 'runs\cd210-mixed-example' `
  --output 'exports\cd210-mixed-example'
```

The output directory must be unused. Export preserves the mixed objective,
recipe, source run and both held-out cohort identities. Local ONNX creation does
not establish device compilation or driving qualification.

## Deployment status

Both grounded parked-car captures completed physical QA. The first capture,
`20261015T145345-4b84e784`, produced 40 candidate labels with observed departure,
pass, return and nominal recovery coverage. Its admission report could not be
saved because a NumPy boolean escaped the coverage calculation. Phase bounds
and predicates now become native JSON numbers and booleans; the acceptance
limits are unchanged. Re-admission reuses retained raw recordings and reruns
independent checks before publication.

The repaired admission published **40 unmarked-road training labels** from that
capture, with no remaining checks or errors, including verification by the
actual training interpreter. The road-52 pedestrian retry passed live grounding:
its original envelope straddled a 0.152 m curb step, and a measured 0.05 m outward
adjustment reduced all four support gaps below 0.0007 m. Its full recording and
admission are separate from this startup result.

The road-52 parked-car capture produced six candidate labels but lacks observed
departure and passing coverage. Correct serialization does not qualify that
episode. Its capture remains retained for diagnosis.

Weather application reads one consistent weather value and allows up to five
simulator frames for readback to match. Persistent mismatches still reject and
name the differing fields. Stationary pedestrian placement searches outward by
at most 0.6 m from the curb toe for four level supported envelope corners, before
recording. Actual settled support is independently checked with the original
4 cm gap/penetration limits. Vehicles retain their original fully-in-road
placement and physical settling.

Collection and continuation now return exit code 2 when any attempted job fails,
including a previously failed skipped job. Process completion alone is not
reported as successful collection. The two pedestrian jobs are retried under a
new immutable source identity; old captures and diagnostics remain intact.

`reports.start_mixed_when_admitted` keeps the existing ordinary-lane trainer
running until unmarked training and validation cohorts are both admitted. It
then requests the old trainer's normal checkpoint stop and seeds an isolated
mixed follower from its exact immutable best checkpoint. No mixed optimizer
step has occurred yet. Earlier curb-mounted captures are explicitly excluded.
A completed capture, stationary probe or gradient test alone grants no labels.
