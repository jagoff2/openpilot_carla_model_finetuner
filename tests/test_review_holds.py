import json
from pathlib import Path

import pytest
import torch

from lane_dataset.loader import SequenceEpisode
from lane_dataset.review_hold import (ReviewHoldError, active_holds,
    assert_checkpoint_not_held, assert_episode_not_held, checkpoint_review_findings)
from lane_training import production as p
from test_loader import write_fixture
from test_production_training import Tiny, corpus


def registry(root, episode="episode-0", status="held"):
    path = root / "reports/review-holds.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(schema_version=1, episodes={episode:
        dict(status=status, reasons=["visual artifacts require review"])})))
    return path


def test_hold_blocks_existing_archive_cache_then_release_restores_access(tmp_path):
    directory = tmp_path / "accepted/episode-0"
    directory.mkdir(parents=True)
    manifest = write_fixture(directory)
    original = manifest.read_bytes()
    with SequenceEpisode(manifest, require_retention=False) as episode:
        expected = episode.frame(0).copy()
        registry(tmp_path)
        with pytest.raises(ReviewHoldError, match="episode-0"):
            episode.frame(0)  # Already loaded archive and same cached frame.
        with pytest.raises(ReviewHoldError):
            episode.selected_targets()
        with pytest.raises(ReviewHoldError):
            SequenceEpisode(manifest, require_retention=False, verify_hashes=False)
        registry(tmp_path, status="released")
        assert (episode.frame(0) == expected).all()
    assert manifest.read_bytes() == original


def test_discovery_checks_holds_before_accepting_unchanged_known_data(tmp_path):
    corpus(tmp_path)
    rows, _ = p.discover(tmp_path)
    registry(tmp_path)
    with pytest.raises(ReviewHoldError, match="episode-0"):
        p.discover(tmp_path, {r["id"]: r for r in rows})
    registry(tmp_path, status="released")
    assert p.discover(tmp_path, {r["id"]: r for r in rows})[0] == rows


def test_held_known_but_untrained_episode_can_leave_published_index(tmp_path):
    index = corpus(tmp_path)
    rows, _ = p.discover(tmp_path)
    known = {r['id']: r for r in rows}
    registry(tmp_path)
    listing = json.loads(index.read_text())
    listing['episodes'] = listing['episodes'][1:]
    index.write_text(json.dumps(listing))
    assert len(p.discover(tmp_path, known)[0]) == 2
    assert_checkpoint_not_held(tmp_path, dict(counts={}, known=known))
    with pytest.raises(ReviewHoldError):
        assert_checkpoint_not_held(tmp_path, dict(counts={rows[0]['sha256']: 1}, known=known))


def test_windows_case_alias_cannot_bypass_hold(tmp_path):
    registry(tmp_path)
    with pytest.raises(ReviewHoldError):
        assert_episode_not_held(tmp_path / "ACCEPTED/EPISODE-0/manifest.json")


def test_checkpoint_hold_uses_consumed_hashes_not_merely_known_episodes(tmp_path):
    registry(tmp_path)
    state = dict(step=24, counts={"seen-clean": 4, "seen-held": 4}, known={
        "a": dict(id=r"accepted\episode-0\manifest.json", sha256="seen-held"),
        "b": dict(id=r"accepted\episode-1\manifest.json", sha256="seen-clean")})
    assert checkpoint_review_findings(tmp_path, state) == [dict(episode="episode-0",
        manifest_sha256="seen-held", passes=4, reasons=["visual artifacts require review"])]
    with pytest.raises(ReviewHoldError, match="checkpoint preserved"):
        assert_checkpoint_not_held(tmp_path, state)
    state["counts"]["seen-held"] = 0
    assert_checkpoint_not_held(tmp_path, state)
    state["counts"]["unmapped"] = 1
    with pytest.raises(ReviewHoldError, match="lineage cannot be checked"):
        assert_checkpoint_not_held(tmp_path, state)


def test_resume_refuses_held_lineage_before_source_migration_or_new_commit(tmp_path, monkeypatch):
    registry(tmp_path)
    cfg = p.TrainingConfig(corpus=str(tmp_path))
    state = dict(step=24, counts={"digest": 4}, known={"id":
        dict(id="accepted/episode-0/manifest.json", sha256="digest")})
    run = tmp_path / "run"
    run.mkdir()
    pointer = run / "resume.json"
    pointer.write_text('{"preserve":true}')
    monkeypatch.setattr(p, "restore", lambda *_: (Tiny(), None, state))
    monkeypatch.setattr(p, "commit", lambda *_: pytest.fail("must not rewrite checkpoint"))
    with pytest.raises(ReviewHoldError, match="episode-0"):
        p.Trainer(cfg, run)
    assert pointer.read_text() == '{"preserve":true}'


def test_hold_arriving_during_backward_prevents_optimizer_step(tmp_path, monkeypatch):
    corpus(tmp_path, splits=("train",))
    row = p.discover(tmp_path)[0][0]
    cfg = p.TrainingConfig(corpus=str(tmp_path))
    trainer = p.Trainer.__new__(p.Trainer)
    trainer.cfg = cfg
    trainer.model = Tiny()
    trainer.reference = Tiny().requires_grad_(False)
    trainer.state = dict(step=0, counts={}, known={row["id"]: row})
    trainer.optimizer = p.make_optimizer(trainer.model, cfg)
    trainer.status = lambda *args, **kwargs: None
    trainer.progress = lambda *args, **kwargs: None
    class Episode:
        def __init__(self, path):
            self.path = Path(path)
            self.manifest = {"split": "train"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def backward(*args, **kwargs):
        for parameter in trainer.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        registry(tmp_path)
        return torch.zeros(1), {"loss": 1.}
    monkeypatch.setattr(p, "SequenceEpisode", Episode)
    monkeypatch.setattr(p, "backward", backward)
    monkeypatch.setattr(trainer.optimizer, "step", lambda: pytest.fail("held data must not update weights"))
    before = p.parameter_digest(trainer.model)
    with pytest.raises(ReviewHoldError):
        trainer.train_one(row)
    assert p.parameter_digest(trainer.model) == before
    assert trainer.state["step"] == 0


@pytest.mark.parametrize("document", [None, {}, {"schema_version": 1, "episodes": []},
    {"schema_version": 1, "episodes": {"episode-0": {"status": "held", "reasons": []}}},
    {"schema_version": 1, "episodes": {"../escape": {"status": "released"}}}])
def test_malformed_registry_fails_closed(tmp_path, document):
    path = tmp_path / "reports/review-holds.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document))
    with pytest.raises(ReviewHoldError): active_holds(tmp_path)
    with pytest.raises(ReviewHoldError): p.discover(tmp_path)
