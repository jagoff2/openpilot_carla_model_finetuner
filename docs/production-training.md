# CD210 production fine-tuning during collection

The production trainer is `python -m lane_training.production`. It follows
`<dataset_root>\training-index.partial.json` while the two CARLA
collectors run concurrently, then follows the final index when published.
The visible console launcher is `train.ps1`.

This run uses the verified CD210 vision and policy weights, the direct observed
centerline objective, and **real AdamW optimizer steps**. It uses two CPU threads
at below-normal process priority. GPU 0 and GPU 1 both remain assigned to CARLA.
The earlier forward/backward checks were individual checks, not an always-running
training service; repeated dense rollouts remain the main training cost.

## Training and intake

- Vision learning rate: `1e-6`; policy learning rate: `1e-5`.
- FP32 weights and arithmetic, AdamW weight decay `0.01`, gradient norm cap `1`.
- One optimizer step accumulates the supported labels in one complete causal
  episode rollout. Both vision and policy must receive finite, nonzero gradients.
- Four passes per accepted training episode. New episodes take priority over
  replay; replay favors less-used scenario families. Once current episodes have
  four passes, the trainer waits and checks for new accepted data every 10 seconds.
- The original frozen CD210 model generates its own temporal context and anchors
  forward progress. The student generates its own context. Measured speed and
  recorded control commands are not model inputs or geometric loss targets.
- Published manifests are immutable and hashed. Geometry groups cannot appear
  in multiple splits. Validation and test episodes cannot enter optimizer steps.

Accepted frames and training targets remain in the original corpus. Checkpoints
are model artifacts under the dataset project's existing workspace:
`runs/cd210-centerline-production`.

## Checkpoints and restart

`resume.json` identifies the committed checkpoint slot and its SHA-256. Each
checkpoint includes the model, optimizer, update count, consumed-episode counts,
manifest identities, configuration and training source hashes. Two alternating
slots preserve the previous commit until the next pointer is atomically saved.
A failed episode update restores the last commit and records the failing episode.

Running the same launcher resumes this run. An operating-system lock prevents
duplicate trainers. Changed training code or configuration requires an explicit
run migration instead of silently resuming different behavior. To stop cleanly,
create `stop.request` in the run directory; remove it before resuming. A stop is
checked between checkpointed operations, so the current episode can finish.

The original CD210 ONNX files and frozen reference remain unchanged. Training
updates candidate checkpoint files only. No vehicle deployment is performed.

Run `./export-c3x.ps1` for an on-demand ONNX and c3x PKL package selected from the
best available saved model. See [c3x export and offline resume](c3x-export.md).

## Validation and outputs

Validation runs on the held-out validation split before training, after the first
update, every ten updates, when new validation episodes arrive, and before
waiting or completion. It compares the student with the fixed CD210 baseline on
the same supported geometry and records lateral placement, heading, clearance,
progress, and changes in other outputs. Validation uses no gradients or optimizer
updates. The test split is never evaluated by this trainer.

- `status.json`: PID, phase, update count, active episode, resource use and the
  concurrent collector snapshot.
- `events.jsonl`: committed updates, parameter hashes, gradient counts, episode
  losses and validation results.
- `checkpoint-0.pt` / `checkpoint-1.pt`: resumable model and optimizer slots.
- `validation/step-*.json`: immutable validation reports for each evaluation.
- `preserved/checkpoints/<SHA-256>.pt`: permanent independent copies of every
  better-than-stock validated candidate and every former best. There is no
  automatic pruning. Rotating resume slots never share these checkpoint files.
- `preserved/records` and `preserved/promotions`: append-only provenance records.
- `best.json`: hash-bound pointer to a preserved winner. A replacement must
  strictly beat stock and the incumbent on the same current validation cohort,
  with no aggregate regression on any earlier promotion cohort. Ties retain the
  incumbent. Adding validation data requires rescoring the incumbent first.
- `final-exports`: immutable, verified ONNX exports selected through the same best
  candidate logic used by the on-demand exporter. Completion never blindly
  exports the latest training weights. `completion.json` identifies the export.

Existing runs from before this preservation change can be migrated explicitly:

```powershell
python -m lane_training.production --run-dir runs/cd210-centerline-production --migrate-preservation
```

Stop the prior trainer cleanly first with its `stop.request` file, then remove
that request before restarting. Migration archives both resume slots and any
legacy `best.pt` before updating the source binding. It preserves weights,
optimizer, step and sampling counts and requests fresh validation. It refuses
changes outside the persistence/export modules. Legacy best files remain intact.

Finite gradients and improving validation loss do not establish driving release
qualification. A training completion report explicitly records that distinction.
Initial partial-corpus validation is provisional because required scenario and
independent-geometry coverage is still growing.

## Lateral-only model selection

Validation, checkpoint promotion and export use `selection_metric: lateral_geometry_v1`.
The score is lateral centerline loss plus 0.25 times heading loss plus boundary
clearance loss, with the same default component scales as training. `loss` and
`baseline_loss` contain this score. Longitudinal quality does not contribute to
ranking, prior-cohort regression checks or tie-breaking. The optimization objective
still includes the existing forward-progress guard so shortening the predicted
path is discouraged during training; it appears separately as `optimization_loss`
and `progress_guard_loss`. Training logs also expose `lateral_score_before_update`
and `progress_guard_before_update` separately.

An existing run can migrate without resetting weights, optimizer, step or sampling:

```powershell
python -m lane_training.production --run-dir runs/cd210-centerline-production --migrate-lateral-selection
```

First stop its trainer with `stop.request`, wait for `stopped_at_checkpoint`, and
remove the request. Both resume slots are archived before migration. Validation
is rerun, and old incumbents are rescored under the new metric before comparison.
Original model files and historical reports remain intact.
