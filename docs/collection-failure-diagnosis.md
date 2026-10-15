# Collection failure diagnosis, 2026-10-14

The policy-v5 failures inspected at 80 accepted labels do not show a usable lane
corridor being rejected solely for poor longitudinal behavior. Saved camera
evidence and telemetry show missing immediate lane-boundary support:

* Town10HD route `auto-52fa8ea8dfee809ddfb1`: both appearances stopped around
  route metres 54–56. Its first junction spans 53.74–95.00 m. At the first
  stopping rejection, the observed right boundary ends at 58.24–58.74 m,
  leaving about 2.14–2.17 m after the vehicle's front-body reserve. This is less
  than the braking-plus-delay requirement at about 3.49 m/s. The next observed
  boundary segment is beyond a long gap. Removing the stop gate would not make
  this gap valid lane-center supervision.
* Town07 route `auto-eafcb09048989d90b06d`: the approach has no painted center
  boundary. During the early controller failure, the selected left-boundary
  segment starts around 74 m while the car is near 28 m. Applying that distant
  curved segment's tangent as a local constraint produces metres of apparent
  clearance error. Later, the common observed corridor shrinks to roughly
  72.74–74.80 m. More solver iterations cannot supply the missing local boundary.
* The other Town07 warmup failure repeatedly handed control back to physical
  initialization because the pixel-supported controller did not qualify. The
  400-frame limit was the final symptom, not evidence that extending warmup
  or relaxing lateral QA would make the route suitable.

The avoidable scheduling defect was `ORDER BY family_priority,ticks,id`, with
severe hairpins first. This put short, unmarked junction paths ahead of other
families regardless of their boundary support. The catalogue contained 9,405
candidates; the new map-metadata ranking classified 5,134 in its lowest risk
tier, 1,186 as intermediate, and 3,085 as high risk. These are candidate counts,
not independent geometry counts or accepted samples.

The fix prioritizes short unpainted gaps in the actual driven portion, including
physical warmup, then
distributes opportunities across families and the separate test-map pool.
Unpainted curb routes stay eligible. No admission threshold, calibration range,
student input, source seal, existing recording or checkpoint is changed.

The first ranking omitted early warmup; a Town04 route exposed why that matters.
Its warmup crossed an unpainted tight junction at 15–33 m, before the painted
labelled event. The physical initializer deviated over four metres and never
qualified. The final ranking includes that prefix. The counts above describe the
initial event/history ranking, not a guarantee of usable warmup or independent
geometry. Including warmup produces 4,374 low-risk, 1,120 intermediate and 3,911
high-risk candidates in the same catalogue.

The newly selected routes also reproduced two depth-fixture implementation bugs:

* Wide-camera rays started 3 m forward after the visible near road intersection,
  then hit a road collider buried 1.25 m below it. A local surveyed-plane check
  now excludes those rays before any sensor depth is inspected.
* Town05 narrow-camera rays hit unlabelled invisible collision volumes before
  the road. A same-frame semantic Road/RoadLines pixel now permits excluding
  NONE volumes. Named occluders still reject; the nearest remaining road hit
  must pass the unchanged 0.15 m depth agreement check.

Live comparisons at one failed Town03 and two failed Town05 spawn poses reproduced
the old failures and passed with the repair in all six camera checks. The largest
forward-depth residual was 0.000823 m. This establishes the repaired sensor
preflight at those poses, not full-episode admission.

The relevant controller, admission, visibility, storage and scheduling suite
passed 224 tests. Existing recordings finish under their original source version
before the automatic policy-v6 update. Whether a
majority of the newly prioritized routes pass must be measured after capture
and final admission; map metadata alone cannot establish that result.

At 80 accepted labels, the first two policy-v6 jobs were still capturing, with
no control rejections in their progress logs. The unchanged counter at that
point was not a new admission rejection. Previous successful captures took
1,141 to 2,028 seconds before postprocessing for 20 selected labels each.
Their per-frame timings showed 0.23-0.36 seconds in image conversion/packing
and 0.36-0.53 seconds in visibility and control, so GPU activity alone was a
poor measure of collection progress.

Two CPU optimizations preserve the existing evidence contract:

* Run the pinned RGB-to-NV12 producer on even 16-row tiles. Its pointwise
  operations and 2x2 chroma reductions are unchanged, with smaller temporary
  arrays. Random full-resolution and negative-stride RGB/BGR inputs must match
  the original producer byte for byte.
* Iterate visible history indices instead of every unobserved point, retaining
  point identity, evidence ordering, causal cutoffs and source-only visibility.

On an accepted recording, seven paired offline measurements gave median RGB
conversion times of 176 ms before and 74 ms after, and history query times of
34 ms before and 18 ms after. All image bytes and all support fields matched.
These are component measurements, not an end-to-end throughput claim. The
relevant suite passed 230 tests with two existing SLSQP bounds warnings.

`reports.handoff_collector_update` finishes and admits existing captures under
their original source seal before applying a hash-bound source patch and test
report. It then resumes both GPU slots concurrently. A slot that finishes first
waits during this one-time update; the helper does not interrupt its recording.
If installation fails, it restores the prior files and resumes collection.
The capture policy number is unchanged because the optimization changes no
acceptance criterion and should not cause deterministic failures to be retried.

The subsequent acceptance correction changes label allocation, not lateral QA:
twenty valid labels remain required, balanced across all five reviewed phases,
with at least two distinct decisions in each phase. Requiring exactly four in
every phase discarded complete events when one short phase had only two or
three valid candidates despite enough verified labels elsewhere. Candidate
selection, automatic review and interrupted-event recovery use the same rule.

A live Town06 candidate exposed a separate annotation defect. Its selected
junction was straight (maximum event curvature about 0.000002/m); the curve
used to classify the full route began more than 50 m after that event ended.
Numerical heading noise within the junction had been interpreted as a turn.
Intersection proposals now require at least 20 degrees of signed heading
change within the junction, measured from the actual center coordinates.
Cached tasks are screened before dispatch, and admission repeats the check
within the annotated event. Other curved events must have resolved heading
change within their own phase ranges. No hidden future curve can qualify a
straight recorded event.

The misclassified 20-label validation shard was returned intact to quarantine,
removed from the published index and active split plan, with a separate
revocation report. Its original manifest and payloads were preserved. Training
resumed from the prior committed step-8 checkpoint; the best checkpoint and
export artifacts were not changed. The combined regression suite passed 258
tests with the two existing SLSQP bounds warnings.

The automatic queue also offered every non-test geometry to validation before
training. Successful validation publication then correctly prevented that same
geometry from being used for training. This produced 200 validation labels but
only 40 training labels, despite configured targets of 500 and 4,000 respectively.
The cache queue now selects the split with the smallest accepted plus in-flight
fraction of its target, using independent cursors for train, validation and test.
Town05 remains test-only, accepted split assignments remain fixed, and family
quotas and boundary ranking still apply. New geometry is offered to one split,
with two appearances for training. Live worker status includes its split and
family. Twenty-three targeted tests passed, including recovery from the observed
imbalance, concurrent reservations, fixed held-out geometry, and convergence to
an 80/10/10 allocation under successful admissions.
