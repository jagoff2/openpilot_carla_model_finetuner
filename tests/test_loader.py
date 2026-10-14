import json

import numpy as np
import pytest
import torch

from lane_dataset.adapter import expert_trajectory_loss
from lane_dataset.loader import DiagnosticSequence, SequenceEpisode, student_rollout, validate_frame_metadata
from lane_dataset.preprocessing import PACKED_FRAME_SHAPE, reference_module


def metadata(count):
    result = []
    for i in range(count):
        desire = [0] * 8
        desire[1 if 2 <= i < 6 else 0] = 1
        result.append(dict(index=i, frame_id=100 + i, timestamp=5.0 + i * .05,
                           desire=desire, traffic_convention=[1, 0], action_t=[.25, .35]))
    return result


@pytest.mark.parametrize("kind", ["missing", "duplicate", "time", "unordered"])
def test_reject_non_contiguous_frames(kind):
    rows = metadata(8)
    if kind == "missing": rows[3]["frame_id"] += 1
    if kind == "duplicate": rows[3]["frame_id"] -= 1
    if kind == "time": rows[3]["timestamp"] += .003
    if kind == "unordered": rows[3]["index"] = 6
    with pytest.raises(ValueError): validate_frame_metadata(rows)


def write_fixture(root, count=105, complete=True):
    archive = reference_module("frame_archive_v2")
    manifest = dict(schema_version=1, status="accepted", frames_file="frames.jsonl",
                    labels_file="labels.npz", archives={})
    for camera, base in (("narrow", 0), ("wide", 100)):
        path = root / f"{camera}.zip"
        with archive.FrameArchiveWriter(path, dtype=np.uint8, frame_shape=PACKED_FRAME_SHAPE,
                                        expected_count=count) as writer:
            for index in range(count):
                writer.write(index, np.full(PACKED_FRAME_SHAPE, (base + index) % 256, np.uint8))
            writer.finalize(complete=complete)
        manifest["archives"][camera] = dict(path=path.name, sha256=archive.sha256_file(path))
    (root / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in metadata(count)), encoding="utf-8")
    np.savez(root / "labels.npz", label_indices=np.array([100]),
             plan_targets=np.zeros((1, 33, 15), np.float32), plan_mask=np.zeros((1, 33, 15), bool),
             current_speed=np.array([5], np.float32), commanded_curvature=np.array([.04], np.float32),
             lateral_mask=np.ones(1, bool))
    manifest["files"] = {path.name: archive.sha256_file(path) for path in root.iterdir() if path.is_file()}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_archive_round_trip_pairing_recurrent_history_and_backward(tmp_path):
    path = write_fixture(tmp_path)
    observed = {}
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    ticks = 0
    def student(inputs):
        nonlocal ticks
        assert set(inputs) == {"img", "big_img", "desire_pulse", "traffic_convention", "action_t", "features_buffer"}
        assert inputs["img"].dtype == torch.uint8
        if ticks in (2, 4, 6, 100):
            observed[ticks] = {key: value.detach().clone() for key, value in inputs.items()}
        # Feature values encode producing tick; lateral output remains differentiable.
        result = torch.zeros((1, 2580)) + parameter * 0
        result[:, 2062] = parameter
        result[:, 2066:2578] = ticks + 1
        ticks += 1
        return result
    with SequenceEpisode(path, require_retention=False) as episode:
        assert episode.validate_all_frames()[0]["all_frames_validated"]
        predictions, targets = student_rollout(episode, student)
        np.testing.assert_allclose(targets["action_t"], [[.25, .35]])
        loss, _ = expert_trajectory_loss(predictions,
            lateral_targets=torch.from_numpy(targets["lateral_targets"]),
            lateral_mask=torch.from_numpy(targets["lateral_mask"]))
        before = parameter.detach().clone()
        loss.backward()
        assert parameter.grad is not None and parameter.grad != 0
        assert torch.equal(parameter.detach(), before)
    assert ticks == 101
    assert torch.all(observed[4]["img"][0, :6] == 0)
    assert torch.all(observed[4]["img"][0, 6:] == 4)
    assert torch.all(observed[4]["big_img"][0, :6] == 100)
    assert torch.all(observed[4]["big_img"][0, 6:] == 104)
    # Tick 100 buffer = outputs [4,8,...,96], whose values are [5,9,...,97].
    torch.testing.assert_close(observed[100]["features_buffer"][0, :, 0], torch.arange(5, 98, 4).float())
    assert observed[6]["desire_pulse"].sum().item() == 1
    assert observed[6]["desire_pulse"][0, 23, 1].item() == 1


def test_incomplete_archive_quarantined_from_loader(tmp_path):
    path = write_fixture(tmp_path, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        SequenceEpisode(path, require_retention=False)


def test_retention_required_by_default(tmp_path):
    path = write_fixture(tmp_path)
    with pytest.raises(ValueError, match="ten seconds"):
        SequenceEpisode(path)


def test_full_history_and_future_complete_sequence(tmp_path):
    path = write_fixture(tmp_path, count=301)
    with SequenceEpisode(path) as episode:
        assert len(episode) == 301
        assert len(list(episode.iter_frames())) == 301


def test_corrupt_archive_rejected_by_hash(tmp_path):
    path = write_fixture(tmp_path)
    with (tmp_path / "wide.zip").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        SequenceEpisode(path, require_retention=False)


def test_fully_ambiguous_labels_cannot_inflate_accepted_counts(tmp_path):
    path = write_fixture(tmp_path)
    with np.load(tmp_path / "labels.npz") as archive:
        labels = {key: archive[key] for key in archive.files}
    labels["lateral_mask"][:] = False
    np.savez(tmp_path / "labels.npz", **labels)
    manifest = json.loads(path.read_text())
    manifest["files"]["labels.npz"] = reference_module("frame_archive_v2").sha256_file(tmp_path / "labels.npz")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="supported supervision"):
        SequenceEpisode(path, require_retention=False)


@pytest.mark.parametrize("filename", ["frames.jsonl", "labels.npz"])
def test_frame_metadata_and_labels_verified_before_use(tmp_path, filename):
    path = write_fixture(tmp_path)
    if filename == "frames.jsonl":
        rows = [json.loads(line) for line in (tmp_path / filename).read_text().splitlines()]
        rows[100]["action_t"] = [.45, .35]  # Valid shape/value that changes supervision semantics.
        (tmp_path / filename).write_text("\n".join(json.dumps(row) for row in rows))
    else:
        with np.load(tmp_path / filename) as archive:
            labels = {key: archive[key] for key in archive.files}
        labels["commanded_curvature"][:] = .08  # Internally consistent but unauthorized target edit.
        np.savez(tmp_path / filename, **labels)
    with pytest.raises(ValueError, match="shard file hash mismatch"):
        SequenceEpisode(path, require_retention=False)


@pytest.mark.parametrize("missing", [None, "frames.jsonl", "labels.npz", "wide.zip"])
def test_required_hash_inventory_cannot_be_omitted(tmp_path, missing):
    path = write_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    if missing is None:
        del manifest["files"]
    else:
        del manifest["files"][missing]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest.files"):
        SequenceEpisode(path, require_retention=False)


def test_all_declared_files_are_verified_not_only_student_payload(tmp_path):
    path = write_fixture(tmp_path)
    report = tmp_path / "admission-report.json"
    report.write_text('{"passed":true}')
    manifest = json.loads(path.read_text())
    manifest["files"][report.name] = reference_module("frame_archive_v2").sha256_file(report)
    path.write_text(json.dumps(manifest))
    report.write_text('{"passed":false}')
    with pytest.raises(ValueError, match="admission-report.json"):
        SequenceEpisode(path, require_retention=False)


def test_archive_declaration_cannot_disagree_with_inventory(tmp_path):
    path = write_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["archives"]["wide"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="archive hash disagrees"):
        SequenceEpisode(path, require_retention=False)


def test_diagnostic_loading_is_explicit_and_does_not_mark_data_accepted(tmp_path):
    path = write_fixture(tmp_path,count=301)
    manifest = json.loads(path.read_text())
    manifest.update(status="development",accepted_label_count=0)
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    with pytest.raises(ValueError,match="accepted"):
        SequenceEpisode(path)
    with DiagnosticSequence(path) as episode:
        assert len(episode) == 301 and episode.selected_targets()["lateral_mask"].all()
        assert episode.manifest["status"] == "development"
    assert path.read_bytes() == before


def test_diagnostic_mode_preserves_masks_and_hash_checks(tmp_path):
    path = write_fixture(tmp_path,count=301)
    with np.load(tmp_path/"labels.npz") as archive:
        labels = {key:archive[key] for key in archive.files}
    labels["lateral_mask"][:] = False
    np.savez(tmp_path/"labels.npz",**labels)
    manifest = json.loads(path.read_text())
    manifest.update(status="development",accepted_label_count=0)
    manifest["files"]["labels.npz"] = reference_module("frame_archive_v2").sha256_file(tmp_path/"labels.npz")
    path.write_text(json.dumps(manifest))
    with DiagnosticSequence(path) as episode:
        assert not episode.selected_targets()["lateral_mask"].any()
        assert not episode.selected_targets()["plan_mask"].any()
    with (tmp_path/"labels.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError,match="hash mismatch"):
        DiagnosticSequence(path)


def test_diagnostic_loader_rejects_accepted_or_misreported_status(tmp_path):
    path = write_fixture(tmp_path,count=301)
    with pytest.raises(ValueError,match="development"):
        DiagnosticSequence(path)
    manifest = json.loads(path.read_text())
    manifest.update(status="development",accepted_label_count=1)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="zero accepted"):
        DiagnosticSequence(path)


def test_streamed_recording_backward_matches_batch_gradient_and_forwards_suffix(tmp_path):
    from lane_dataset.smoke import recorded_rollout_backward
    path = write_fixture(tmp_path,count=301)
    with np.load(tmp_path/"labels.npz") as archive:
        labels = {key:archive[key] for key in archive.files}
    labels = {key:np.repeat(value,3,axis=0) for key,value in labels.items()}
    labels["label_indices"] = np.array([100,110,120])
    labels["commanded_curvature"] = np.array([.04,.08,.12],np.float32)
    labels["lateral_mask"][1] = False
    np.savez(tmp_path/"labels.npz",**labels)
    manifest = json.loads(path.read_text())
    manifest.update(status="development",accepted_label_count=0)
    manifest["files"]["labels.npz"] = reference_module("frame_archive_v2").sha256_file(tmp_path/"labels.npz")
    path.write_text(json.dumps(manifest))
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.5))
            self.ticks = 0
        def forward(self,inputs):
            self.ticks += 1
            values = torch.arange(2580,dtype=torch.float32).reshape(1,-1)/2580
            return values*self.weight
    batch, streamed = Student(), Student()
    with DiagnosticSequence(path,require_retention=False) as episode:
        predictions, targets = student_rollout(episode,batch,stop_after_last_label=False)
        loss, _ = expert_trajectory_loss(predictions,
            lateral_targets=torch.from_numpy(targets["lateral_targets"]),
            lateral_mask=torch.from_numpy(targets["lateral_mask"]))
        loss.backward()
        output, _, calls = recorded_rollout_backward(episode,streamed)
    torch.testing.assert_close(streamed.weight.grad,batch.weight.grad)
    torch.testing.assert_close(output,predictions.detach())
    assert streamed.ticks == batch.ticks == 301 and calls == 2
    assert streamed.weight.item() == .5 and not output.requires_grad
