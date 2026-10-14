# Optional surrounding traffic

`lane_dataset.traffic` supplies 2–4 physics-driven vehicles in adjacent, same-direction driving lanes. Spawn candidates exclude junctions, opposite directions, and non-driving lanes. Actual bounding boxes must clear the complete intended center corridor and remain at least 8 m from the ego and other traffic bodies. If the requested count cannot be placed safely, setup fails; it does not silently reduce the traffic count.

Traffic is now integrated into `capture.py`, the CLI, the benchmark orchestrator, and `launch.ps1`. Enable two neighboring vehicles with `-Traffic 2`; 3 and 4 are also supported, and 0 disables the option. Benchmark and development commands also take `-Speed 2.5`. Bulk generation uses each task's `traffic_count` and `nominal_speed_mps`; launcher development overrides do not rewrite planned tasks.

```powershell
.\launch.ps1 -Mode Develop -Workers 1 -Ticks 600 -Speed 2.5 -Traffic 2 -DatasetRoot '<dataset_root>'
```

Capture creates the helper after spawning the ego and calls it on every owned world snapshot before receiving the corresponding camera images, including physical warmup. Final cleanup drains collision callbacks, adds cleanup errors, and writes `traffic-report.json` into the shard's file-hash manifest. Any failed live check or final summary quarantines the capture. Admission checks the bound report against the manifest, requested/spawned counts, configured physics, clearances, and the sum of warmup plus recorded frames. The aggregate report is retained; capture does not currently save every surrounding actor pose for a separate offline replay.

For a separate world-owner integration, the helper API is:

```python
from lane_dataset.traffic import SurroundingTraffic, TrafficConfig

traffic = SurroundingTraffic.spawn(
    client, world, ego, full_validation_reference,
    worker=worker, rpc_port=port, seed=episode_seed,
    config=TrafficConfig(count=2, speed_mps=2.5),
)
# After every owner-controlled world tick, including warmup:
traffic_frame = traffic.check(snapshot)
# Optional callers can retain traffic_frame as simulator-only QA evidence.
# In finally, before destroying the ego:
cleanup_errors = traffic.close()
traffic_summary = traffic.summary()
```

`check` uses transforms from that exact world snapshot. It raises `TrafficRejected`, carrying a `.report`, on any corridor intrusion, insufficient body clearance, collision event, missing actor, missing frame, or non-20-Hz sequence. Failure reasons remain latched. `close` never ticks, destroys only helper-owned traffic and collision sensors, disables its Traffic Manager synchrony, and returns cleanup errors. Capture must preserve the report and quarantine on any exception, cleanup error, or failed final summary. Keep this helper inside the existing `world_owner` context and create only one Traffic Manager per world. Ports are fixed to worker 0: RPC 2400/TM 8400 and worker 1: RPC 2500/TM 8500.

The helper explicitly enables vehicle physics, disables hybrid physics and dormant respawning, seeds the Traffic Manager, disables automatic and random lane changes, and retains collision/light/sign/walker handling. The capture process remains the sole tick owner. These settings follow CARLA's synchronous and deterministic Traffic Manager requirements. [CARLA 0.9.16 Traffic Manager](https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/)

Desired speed is specified to the helper in m/s and converted to the Traffic Manager's km/h input; CARLA's motion planner divides the configured target velocity by 3.6. [CARLA 0.9.16 motion planner source](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/trafficmanager/MotionPlanStage.cpp#L124)

The obstruction check protects a centerline corridor with half-width equal to half the ego's nominal body width plus 15 cm. Full lane geometry is used only for spawning and independent obstruction validation, never to form expert targets or student inputs. The center spline is conservatively enclosed by a sampled polyline plus a second-derivative chord-error bound. Traffic footprints use all eight 3D bounding-box vertices projected into XY, preserving roll and pitch effects. Between-frame envelopes include the convex hull of both endpoint footprints plus a rotational chord-error bound, assuming linear translation and shortest rotation between 20-Hz snapshots. This is not a physics-substep trace. The XY check conservatively rejects vertically separated overlaps. Exact body-to-body clearance against the recorded ego is also checked. Complete swept ego lane-boundary qualification remains a separate gate.

Traffic Manager vehicles can select junction paths or drift despite disabled lane changes. Such behavior causes rejection when it obstructs the protected corridor; Traffic Manager output is never accepted as expert behavior. Sparse collision events have no per-frame empty-event acknowledgement. Current capture checks each snapshot and drains the event queue again after stopping the sensors; it does not claim an independently acknowledged empty collision stream for each frame. A live spawn/tick probe establishes helper execution only. Actual traffic appearance coverage, image evidence, and accepted-label quality require integrated recording and admission evidence.

The standalone bounded probe is `reports/traffic-probe.py`. Run it only after port 2400 has been released by other work. It takes the world-owner lock, loads Town04, spawns two vehicles, measures 80 physics ticks, and writes an immutable timestamped report under the dataset's `reports/traffic-probe-*` directory. It never creates or publishes a training shard.

The recorded standalone probe completed all 80 ticks with no collision events, minimum corridor clearance 1.18645 m, and no cleanup errors. [Probe evidence and revision limits](../reports/traffic-helper-validation.md) distinguish this execution from subsequent helper refinements. The 600-frame recorded-sequence checkpoint check used an earlier traffic-free recording with RGB postprocessing disabled. Current RGB postprocessing and explicit instantaneous pinhole settings require fresh camera evidence, as described in the [dataset card](../DATASET_CARD.md). No accepted traffic labels or final worker-comparison result is inferred from either check.
