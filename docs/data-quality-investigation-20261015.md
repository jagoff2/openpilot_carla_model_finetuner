# Recording and training quality repair, 2026-10-15

Collection and training are stopped. Existing accepted recordings, checkpoints,
and model exports have been preserved. This investigation did not update model
weights or install a new model on the device.

## Findings backed by saved data

The three reported recordings are `straight_recovery` episodes. They hold the
vehicle 0.25 m right of the centerline for the five-second recurrent history,
then command a physical return to center. All supervised labels begin at or
after that release. Measured convergence takes 1.75-1.8 seconds, with at most
1.78 cm overshoot. Their geometric training targets use the observed boundary
midpoint, not the held offset or recorded steering path.

A cache bug copied the positive recovery offset into the second appearance,
although task preparation intended opposite signs. This affected three accepted
appearance-1 recoveries. Cached queue expansion and worker preparation now both
rebind the offset sign. New review videos show the measured center offset and
explicitly identify the unlabelled hold and subsequent recovery.

All 22 accepted recordings were audited: 44 lossless camera archives and 38,730
camera frames passed archive, member, and decoded-payload hashes. The largest
measured center offset outside declared recovery periods was 1.79 cm. This
establishes file integrity and measured vehicle placement; it does not establish
the visual quality of every rendered pixel.

The visible blocky wet reflections and streaks already occur in the original
full-resolution RGB audit images. A stationary comparison at the recorded
Town06 pose reproduced them using WetCloudySunset with precipitation 25,
precipitation deposits 50, and wetness 70. Removing rain, deposits, and wetness
removed those defects at the same pose. Production appearance 1 now uses dry
cloudy daylight. Requested and returned weather values are recorded and checked.

Admission previously checked boundary pixels, packing, and hashes without a
whole-camera blank/frozen-image check. Capture policy 8 adds those checks for
every retained packed frame and requires the dry rendering profile. A sustained
identical camera payload while the car moves is rejected; a stationary tail is
allowed. This is not a general classifier for arbitrary rendering artifacts.
Legacy wet automatic captures require explicit review before readmission.

Recovery QA now bounds the unlabelled prefix and recovery exemption: the prefix
must hold its signed initial offset within 5 cm, and outward excursion or center
overshoot during recovery cannot exceed 5 cm. Admission also binds the declared
sign, release frame, release timestamp, measured offset, and label start. Ordinary
center tracking and physical acceleration/jerk limits remain in force.

## Corrected training objective

An independent defect existed in the direct centerline loss. For an off-center
car, it preferred a path whose first point was already translated onto the lane
center. The corrected objective anchors position and heading at the ego vehicle,
uses a fixed smooth geometric recovery, then follows the exact observed centerline.
It retains the original model's progress reference and boundary-clearance penalty.
See [the objective contract](direct-centerline-objective.md) for its explicit
geometric bounds and limitations.

On the actual reported Town06 corridor at frame 100, the corrected 5.5 m recovery
scores 0.00000336 while an impossible sideways-translated path scores 2.48437.
The gradient moves the translated initial point back toward the ego origin.
Across all 440 archived labels, 439 support this objective. One validation label
has only 2.177 m of continuous observed geometry for a heading recovery requiring
approximately 3 m; it is explicitly masked. All 180 archived training labels have
sufficient geometry, before applying external review holds.

The objective and selection metric are versioned. Existing training runs cannot
silently resume or migrate into the changed objective, and their old loss values
are not comparable with new-objective scores.

## Data holds and checkpoint consequences

`<corpus>/reports/review-holds.json` currently holds the four wet-weather
recordings. The reported dry clip was released after review confirmed the declared
recovery, clean dry RGB, and valid corrected targets. The training index excludes
80 labels. There are 360 unheld labels: 100 training, 200 validation, and 60 test. All 440 original labels remain on disk.
Accepted directories and their manifests have not been edited or moved.

The loader checks holds before loading and while reading frames. The trainer
checks committed training lineage, and checks again before an optimizer update.
Exports also reject checkpoints trained on held recordings. A malformed hold
registry fails closed. Holds are reversible by changing an entry's status to
`released` after review; changing the index alone cannot bypass them.

All currently saved trained checkpoints include at least one held wet recording.
Step 24 is included, as is its existing device candidate. The original CD210
source checkpoint is unaffected. No saved trained checkpoint is claimed as an
unexposed fallback. Historical files and best pointers remain preserved.

The original 200-label validation cohort improved through step 24, then worsened
at steps 28 and 32. The large loss at step 33 came from a different intersection
training episode. Those observations do not establish that wet images caused the
loss increase. They also do not validate the old origin-defective objective.

## Evidence and next run

The workstation evidence is under
`G:\lane-centering-dataset-v1\reports\corruption-investigation-20261015`, with
additional audit/test reports under the project directory of the same name.
It includes the original/dry render comparisons, actual weather readback,
whole-camera payload audit, checkpoint lineage, objective audit, and regression
results. The archive/trajectory audit can be repeated using
`python -m reports.audit_capture_integrity` (see its command-line help).

No full production collection or optimizer run has been restarted after these
changes. A future clean training run must start from the original CD210 with the
new objective and the filtered corpus index, preserving the old run. New captured
data must pass policy-8 admission. Dry daylight is the current rendering scope;
wet-weather diversity requires separate rendering qualification.

Integrated regression result: 217 tests passed in the live project; 53 focused
portability tests passed after synchronizing the packaged repository.
