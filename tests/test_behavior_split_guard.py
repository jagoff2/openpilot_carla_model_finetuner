from contextlib import contextmanager
from dataclasses import asdict
import errno

import pytest

from lane_dataset.config import sha256_file,write_json
from lane_dataset.splits import GeometrySection,RoadInterval
from lane_behavior.loader import SCHEMA,ADMISSION_SCHEMA,REQUIRED_CHECKS
from lane_behavior.split_guard import supplemental_split_errors,assert_supplemental_split,publication_lock


def section(identity="source",town="Town02",road="1",start=0.,end=100.,**kwargs):
    return GeometrySection(identity,town,"straight_recovery",(RoadInterval(road,start,end),),**kwargs)


def published(root,geometry,*,split="validation",name="behavior-existing"):
    directory=root/"accepted"/name;directory.mkdir(parents=True)
    write_json(directory/"behavior-split.json",dict(passed=True,split=split,candidate_section=asdict(geometry)))
    split_sha=sha256_file(directory/"behavior-split.json")
    write_json(directory/"admission.json",dict(schema=ADMISSION_SCHEMA,passed=True,
        method="independent_raw_evidence_and_physical_replay",checks={key:True for key in REQUIRED_CHECKS},
        input_files={"behavior-split.json":split_sha}))
    write_json(directory/"behavior-manifest.json",dict(schema=SCHEMA,status="accepted",training_ready=True,
        capture_scope="production",accepted_label_count=1,split=split,admission_file="admission.json",
        files={"behavior-split.json":split_sha,"admission.json":sha256_file(directory/"admission.json")}))
    return directory


def test_legacy_candidate_cannot_reuse_behavior_validation_road_under_new_ids(tmp_path):
    published(tmp_path,section())
    candidate=section("different-id",start=90.,end=10.,direction="left")
    errors=supplemental_split_errors(tmp_path,section=asdict(candidate),split="train")
    assert errors and any("geometry_leakage" in e for e in errors)
    assert not supplemental_split_errors(tmp_path,section=candidate,split="validation")
    with pytest.raises(ValueError,match="geometry_leakage"):
        assert_supplemental_split(tmp_path,section=candidate,split="train")


def test_native_namespaces_preview_margin_and_town05_policy_are_preserved(tmp_path):
    published(tmp_path,section())
    assert not supplemental_split_errors(tmp_path,section=section(town="Town03"),split="train")
    assert supplemental_split_errors(tmp_path,section=section(start=104.9,end=120.),split="train")
    assert not supplemental_split_errors(tmp_path,section=section(start=105.1,end=120.),split="train")
    assert supplemental_split_errors(tmp_path,section=section(town="Town05"),split="train")


def test_connected_junctions_cannot_hide_cross_split_geometry(tmp_path):
    published(tmp_path,section(road="1",junction_ids=("8",),junction_links=(("8","9"),)))
    candidate=section("candidate",road="42",junction_ids=("9",))
    assert supplemental_split_errors(tmp_path,section=candidate,split="train")


@pytest.mark.parametrize("change",["split","admission","manifest"])
def test_changed_supplemental_binding_fails_closed(tmp_path,change):
    directory=published(tmp_path,section())
    file={"split":"behavior-split.json","admission":"admission.json","manifest":"behavior-manifest.json"}[change]
    (directory/file).write_text("{}")
    errors=supplemental_split_errors(tmp_path,section=section(road="other"),split="train")
    assert errors and errors[0].startswith("supplemental_split_inventory_invalid")


def test_shared_lock_waits_for_busy_owner_and_does_not_retry_publication_errors(tmp_path,monkeypatch):
    import lane_behavior.split_guard as module
    calls=[];released=[];clock=[0.]
    @contextmanager
    def owner(name,root):
        calls.append(name)
        if len(calls)<3:raise OSError(errno.EACCES,"busy")
        try:yield
        finally:released.append(True)
    monkeypatch.setattr(module,"world_owner",owner)
    monkeypatch.setattr(module.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(module.time,"sleep",lambda delay:clock.__setitem__(0,clock[0]+delay))
    with pytest.raises(OSError,match="actual commit error"):
        with publication_lock(tmp_path):
            raise OSError(errno.EACCES,"actual commit error")
    assert calls==["corpus-publication"]*3 and released==[True] and clock[0]==.5


def test_shared_publication_lock_has_finite_timeout(tmp_path,monkeypatch):
    import lane_behavior.split_guard as module
    clock=[0.]
    @contextmanager
    def owner(*args):
        raise OSError(errno.EACCES,"busy")
        yield
    monkeypatch.setattr(module,"world_owner",owner)
    monkeypatch.setattr(module.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(module.time,"sleep",lambda delay:clock.__setitem__(0,clock[0]+delay))
    with pytest.raises(TimeoutError):
        with publication_lock(tmp_path,timeout_s=.5):pass
    assert clock[0]==.5
    with pytest.raises(ValueError):
        with publication_lock(tmp_path,timeout_s=601):pass
