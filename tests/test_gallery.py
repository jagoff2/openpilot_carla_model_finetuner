import base64
import json
import hashlib
import os

import pytest

from lane_dataset.gallery import build_gallery,validate_review_dispositions


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j9t8AAAAASUVORK5CYII=")


def manifest_fixture(tmp_path):
    episode = tmp_path / "quarantine" / "episode-0"
    audit = episode / "audit"
    audit.mkdir(parents=True)
    (audit / "wide-00000.png").write_bytes(PNG)
    manifest = {"geometry_group_id":"geom-0","episode_id":"episode-0","family":"constant_radius",
                "audit_frames":[{"frame_id":123,"timestamp_s":5.,"stage":"entry",
                                 "files":{"wide":"audit/wide-00000.png"}}]}
    (episode / "manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    return episode


def test_actual_images_embedded_and_reviews_stay_pending(tmp_path):
    manifest_fixture(tmp_path)
    report = build_gallery(tmp_path)
    assert report["geometry_groups"] == 1 and report["embedded_images"] == 1
    assert report["reviewed_geometry_groups"] == 0
    template = json.loads((tmp_path / "reports/review-gallery.review-template.json").read_text())
    assert template["geometry_reviews"][0]["status"] == "pending"
    assert "data:image/png;base64," in (tmp_path / "reports/review-gallery.html").read_text()
    assert not validate_review_dispositions(tmp_path,report["review_template_path"])["qualified"]


def test_gallery_rejects_external_audit_path(tmp_path):
    episode = manifest_fixture(tmp_path)
    manifest = json.loads((episode / "manifest.json").read_text())
    manifest["audit_frames"][0]["files"]["wide"] = "../../../../external.png"
    (episode / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="escapes"):
        build_gallery(tmp_path)


def test_missing_continuous_clip_prevents_review_pass(tmp_path):
    manifest_fixture(tmp_path)
    report = build_gallery(tmp_path)
    path = tmp_path / "reports/review-gallery.review-template.json"
    review = json.loads(path.read_text())
    row = review["geometry_reviews"][0]
    row.update(status="pass",reviewer="reviewer",entry_reviewed=True,apex_reviewed=True,
               exit_reviewed=True,crop_loss_intervals_reviewed=True,continuous_clip_reviewed=True)
    path.write_text(json.dumps(review))
    result = validate_review_dispositions(tmp_path,path)
    assert result["errors"] == ["missing_continuous_review_clip:geom-0"]
    assert not result["qualified"]


def test_missing_audit_image_reported_and_no_placeholder_created(tmp_path):
    episode = manifest_fixture(tmp_path)
    (episode / "audit/wide-00000.png").unlink()
    report = build_gallery(tmp_path)
    assert report["embedded_images"] == 0 and len(report["missing_audit_files"]) == 1


def test_representative_prefers_completed_bound_clip_over_failed_or_unbound_capture(tmp_path):
    old = manifest_fixture(tmp_path)
    base = json.loads((old / "manifest.json").read_text())
    failed = {**base, "error": {"message": "interrupted"}, "audit_frames": base["audit_frames"]*10}
    (old / "manifest.json").write_text(json.dumps(failed))
    for name, digest in (("completed-600", hashlib.sha256(b"review clip").hexdigest()), ("newer-unbound", "wrong")):
        episode = tmp_path / "quarantine" / name
        (episode / "audit").mkdir(parents=True)
        (episode / "audit/wide-00000.png").write_bytes(PNG)
        (episode / "review.mp4").write_bytes(b"review clip")
        manifest = {**base, "episode_id": name, "measured_throughput": {"ticks": 600},
                    "review_clip": "review.mp4", "files": {"review.mp4": digest}}
        (episode / "manifest.json").write_text(json.dumps(manifest))
    report = build_gallery(tmp_path)
    review = json.loads(open(report["review_template_path"]).read())["geometry_reviews"][0]
    assert review["episode_id"] == "completed-600"
    assert review["status"] == "pending"


def test_representative_prefers_latest_equal_capture_but_accepted_still_first(tmp_path):
    from lane_dataset.gallery import _representative_rank
    old = manifest_fixture(tmp_path)
    manifest = json.loads((old / "manifest.json").read_text())
    manifest["measured_throughput"] = {"ticks": 600}
    newer = tmp_path / "quarantine/newer"
    newer.mkdir()
    (newer / "manifest.json").write_text(json.dumps(manifest))
    os.utime(old / "manifest.json", ns=(1000000000, 1000000000))
    os.utime(newer / "manifest.json", ns=(2000000000, 2000000000))
    assert _representative_rank((newer / "manifest.json", manifest), tmp_path) < _representative_rank((old / "manifest.json", manifest), tmp_path)
    accepted = tmp_path / "accepted/example"
    accepted.mkdir(parents=True)
    (accepted / "manifest.json").write_text(json.dumps(manifest))
    assert _representative_rank((accepted / "manifest.json", manifest), tmp_path) < _representative_rank((newer / "manifest.json", manifest), tmp_path)
