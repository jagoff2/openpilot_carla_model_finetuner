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
- `validation-step-*.json`: validation results for that accepted validation set.
- `best.pt` / `best.json`: a candidate that improves on original CD210's validation
  centerline loss. Scores from different validation cohorts are not ranked together.
- `final-onnx`: the final training-state exports once collection publishes a
  complete corpus and all training episodes reach four passes without failures.

Finite gradients and improving validation loss do not establish driving release
qualification. A training completion report explicitly records that distinction.
Initial partial-corpus validation is provisional because required scenario and
independent-geometry coverage is still growing.
