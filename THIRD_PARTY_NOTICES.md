# Source and dependency notices

The root MIT license is retained from this repository's initial commit. It does
not replace notices or terms attached to third-party files or downloaded assets.

`lane_reference/runtime/drive_helpers.py`, `driving_action.py`, and
`latcontrol_angle.py` contain verbatim source definitions used for independent
interface tests. Their IQ.Lvbs notices are preserved. The corresponding license
is reproduced in [licenses/IQPilot.txt](licenses/IQPilot.txt). `vehicle_model.py`
preserves the source definition and documentation from the installed
IQDBC/openpilot vehicle-model implementation. Upstream openpilot's MIT notice is
included in that license text. Source hashes and extracted definitions appear in
[lane_reference/PROVENANCE.json](lane_reference/PROVENANCE.json).

Other bundled references are the local camera, archive, RGB/NV12 and diagnostic
implementations used by this pipeline, with provenance in the same manifest.
Historical class names in those helpers do not select or download a checkpoint.

CARLA, its client wheel, maps and binaries are separately installed. PyTorch,
ONNX, ONNX Runtime, NumPy, SciPy, Pillow, psutil and imageio-ffmpeg are installed
dependencies under their upstream licenses. None of their binaries is committed.

CD210 weights come from comma's pinned public openpilot release and are not
distributed in Git. Native artifacts are compiled using the runtime installed
on the c3x. This repository does not redistribute IQPilot's complete runtime or
firmware. Downloaded models and installed runtimes retain their upstream terms.
