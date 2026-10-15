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

The fix prioritizes short unpainted gaps in the actual driven portion, then
distributes opportunities across families and the separate test-map pool.
Unpainted curb routes stay eligible. No admission threshold, calibration range,
student input, source seal, existing recording or checkpoint is changed.

Eighteen targeted scheduling, failure-memory and parallel-worker tests passed.
The existing recordings continue through the supervisor update. Whether a
majority of the newly prioritized routes pass must be measured after capture
and final admission; map metadata alone cannot establish that result.
