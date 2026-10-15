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
