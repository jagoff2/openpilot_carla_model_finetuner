# Rejection investigation, 2026-10-15

This investigation used retained telemetry, raw boundary cross-sections, and a
diagnostic CARLA replay of failed warmup poses. Replays are diagnostic results,
not newly admitted corpus labels.

| Cause | Measured evidence | Correction |
| --- | --- | --- |
| Intersecting paint mistaken for one lane stripe | Town04 event `auto-1e2a0fee8aa9489a8ad3`: declared 0.15 m stripe measured as 0.235–0.265 m merged paint. Its midpoint falsely produced 0.0675 m disagreement. Replaying 944 frames with the width check gives 0.025 m maximum disagreement and passes. A second event, `auto-b0f96a7d4d67ca024abd`, changes from 0.0775 m to 0.0225 m over 1,059 frames. | Reject unresolved widths before comparing position. The check applies equally to centered and displaced paint. Correct-width stripes displaced more than 5 cm still fail. |
| Ambiguous progress on looping routes | Town03 event `auto-a18d396f8b6a8a5e9aee` jumps 249.627 m between consecutive warmup frames. Continuous projection changes 198 of 400 frames and limits the largest step to 0.157 m. | Seed projection from the route start, then use previous progress in capture and physical initialization. |
| Boundary unavailable in model crops | Replay of Town03 warmup poses 0, 200, 399 finds 45, 36, 38 visible left-boundary points in the full wide image, but zero in either model crop at every sampled pose. | Keep this visibility rejection. Preserve failure memory across policy 6 to 7 for unchanged warmup/depth failures instead of automatically repeating every failed route. Identical failed warmup corridors are shared across route IDs. |
| Completed events exceed lateral jerk limit | One complete recovery records 5.0299 m/s³ at 3.5 m/s. Other complete intersections have jerk failures in their stopping tails despite small center offsets. | Offer one 2.5 m/s recapture using an already qualified measured calibration. Rebudget recording duration; preserve phase geometry and physical limits. Normal split, overlap, visibility, and admission checks still apply. A retry is not a promised acceptance. |
| Supervisor discards measured-result metadata | Publication replaces the worker result with a short status dictionary, losing capture-policy version and label count. This prevents immediate failure caching and some targeted retries. | Retain both fields when publication produces the final disposition; regression test exercises an actual publication rejection. |

The physical limits remain 3.5 m/s² lateral acceleration, 5 m/s³ lateral jerk,
and 5 cm rendered-boundary disagreement. No accepted manifest, captured source
seal, checkpoint, or model export is rewritten by this change.

Policy 7 retries corrected paint failures and applicable controller failures.
It does not blanket-retry policy-6 warmup or depth failures. Lower-speed retries
have distinct job IDs and speed-specific failure memory. Accepted assignments
and Town05 test isolation remain enforced. One proposed slower recovery was
correctly screened out because its road overlaps an already accepted group.

165 targeted tests passed, covering capture, initialization, rendered boundaries,
retry memory, concurrent scheduling, preprocessing, controller behavior, split
isolation, and the qualification-required runtime tests.

Read-only boundary/progress replay:

```powershell
python -m reports.audit_rejection_causes --root G:\lane-centering-dataset-v1 `
  --job auto-1e2a0fee8aa9489a8ad3-validation-0 `
  --job auto-b0f96a7d4d67ca024abd-train-0 `
  --warmup-shard G:\lane-centering-dataset-v1\quarantine\auto-a18d396f8b6a8a5e9aee-validation-0-a1-08ea794587 `
  --output G:\lane-centering-dataset-v1\reports\rejection-causes-v7.json
```

`reports.probe_failed_warmup` additionally replays saved poses in CARLA. It waits
for a free simulator port, temporarily pauses only the collector supervisor to
prevent new dispatch onto that port, and resumes the supervisor in `finally`.
Other active capture processes continue. It writes diagnostic images and sensor
evidence only. Do not run it concurrently with a supervisor handoff.

Deployment drains and admits existing captures under their original source
version, installs a hash-bound patch and passing test report, then resumes both
GPU collection slots. CPU training is independent of that handoff.
