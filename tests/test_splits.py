from dataclasses import replace

import pytest

from lane_dataset.splits import (FAMILIES, GeometryGroup, GeometrySection, RoadInterval, audit_crop_loss_evidence,
    SplitAssignment, assign_splits, audit_release, audit_split_overlap, group_geometry_sections)


def section(name, *, town="Town03", road="1", start=0., end=100., family=FAMILIES[0], **kwargs):
    return GeometrySection(name,town,family,(RoadInterval(road,start,end),),**kwargs)


def test_overlap_reverse_and_appearance_share_one_group():
    sections = [section("forward",start=0,end=100),section("reverse",start=80,end=20),
                section("wet_variant",start=30,end=110),section("separate",road="2")]
    groups = group_geometry_sections(sections)
    assert sorted(len(group.sections) for group in groups) == [1,3]
    assert group_geometry_sections(list(reversed(sections))) == groups


def test_adjacent_preview_margin_and_connected_junctions_are_grouped():
    sections = [section("a",road="1",junction_ids=("1",),junction_links=(("1","2"),)),
                section("b",road="2",junction_ids=("2",)),
                section("c",road="3",start=0,end=100),
                section("d",road="3",start=104,end=130)]
    assert sorted(len(group.sections) for group in group_geometry_sections(sections)) == [2,2]


def test_road_ids_are_namespaced_by_town():
    assert len(group_geometry_sections([section("a"),section("b",town="Town05")])) == 2


def test_procedural_parameter_duplicates_cannot_cross_splits():
    one = GeometrySection("proc1","OpenDrive",FAMILIES[3],procedural=True,
                          procedural_parameters={"radius_m":15.,"angle_rad":3.1})
    two = replace(one,section_id="proc2",town="OpenDrive_other")
    assert len(group_geometry_sections([one,two])) == 1
    forged = [GeometryGroup("a",(one,)),GeometryGroup("b",(two,))]
    assignments = [SplitAssignment("a","train",FAMILIES[3],2,0),
                   SplitAssignment("b","test",FAMILIES[3],1,1)]
    assert any(error.startswith("geometry_leakage") for error in audit_split_overlap(forged,assignments))


def test_procedural_mirror_split_and_family_annotations_cannot_disguise_reuse():
    one=GeometrySection("one","OpenDrive",FAMILIES[3],procedural=True,
        procedural_parameters={"family":"hairpin","radius_m":15.,"turn":"left","split":"train"})
    two=GeometrySection("two","OpenDrive",FAMILIES[1],procedural=True,
        procedural_parameters={"family":"hairpin","radius_m":15.,"turn":"right","split":"test"})
    assert len(group_geometry_sections([one,two]))==1


def test_junction_ids_normalize_across_api_and_json_representations():
    sections=[section("one",junction_ids=(1,),junction_links=((1,2),)),
              section("two",road="2",junction_ids=("2",))]
    assert sections[0].junction_ids==("1",)
    assert len(group_geometry_sections(sections))==1


def test_procedural_flag_cannot_make_town05_group_eligible_for_training():
    groups=[]
    for family in FAMILIES:
        for town in ("Town03","Town04","Town05"):
            native=section(f"{family}-{town}",town=town,road=f"{family}-{town}",family=family)
            if family==FAMILIES[0] and town=="Town03":
                native=replace(native,town="Town05")
                procedural=GeometrySection("proc","OpenDrive",family,procedural=True,
                    procedural_parameters={"family":"line","length_m":100.})
                groups.append(GeometryGroup(f"{family}-{town}",(native,procedural)))
            else:
                groups.append(GeometryGroup(f"{family}-{town}",(native,)))
    with pytest.raises(ValueError,match="insufficient independent"):
        assign_splits(groups,group_quotas={"train":5,"validation":5,"test":5})


def test_recomputed_overlap_catches_manipulated_group_ids():
    groups = [GeometryGroup("a",(section("a"),)),GeometryGroup("b",(section("b",start=10,end=20),))]
    assignments = [SplitAssignment("a","train",FAMILIES[0],2,0),SplitAssignment("b","validation",FAMILIES[0],1,1)]
    assert any(error.startswith("geometry_leakage") for error in audit_split_overlap(groups,assignments))


def inventory():
    sections = []
    for family in FAMILIES:
        for index in range(30):
            town = "Town05" if index >= 25 else "Town03"
            sections.append(section(f"{family}-{index}",town=town,road=f"{family}-{index}",family=family))
    return group_geometry_sections(sections)


def test_seeded_assignment_exact_group_quotas_and_town05_test_only():
    groups = inventory()
    assignments = assign_splits(groups)
    assert assignments == assign_splits(list(reversed(groups)))
    assert len({a.group_id for a in assignments}) == 150
    assert {split:sum(a.split==split for a in assignments) for split in ("train","validation","test")} == {
        "train":100,"validation":25,"test":25}
    assert not audit_split_overlap(groups,assignments)


def test_assignment_insufficient_coverage_fails_without_quota_relaxation():
    with pytest.raises(ValueError,match="quotas remain unchanged"):
        assign_splits(inventory()[:-1])


def test_town05_cannot_be_assigned_to_training():
    groups = group_geometry_sections([section("town05",town="Town05")])
    assignments = [SplitAssignment(groups[0].group_id,"train",FAMILIES[0],2,0)]
    assert any(error.startswith("town05_outside_test") for error in audit_split_overlap(groups,assignments))


def complete_records():
    """Synthetic accounting fixture only; never exported as captured samples."""
    groups = inventory()
    assignments = assign_splits(groups)
    records = []
    family_indices = {}
    for assignment in assignments:
        key = (assignment.split,assignment.family)
        index = family_indices.get(key,0)
        family_indices[key] = index+1
        direction = "straight" if assignment.family == FAMILIES[0] else "left" if index % 2 == 0 else "right"
        for appearance in range(assignment.appearances):
            for decision in range(20):
                timestamp = 5.+.5*decision
                refs = [{"frame_id":frame,"timestamp_s":timestamp-age,"camera":"wide"}
                        for frame,age in [(0,.20),(1,.15),(2,.10)]]
                visibility = {"category":"history_crop_loss","point_ids":["near","curve"],
                    "distance_m":[5.,20.],"required_mask":[True,True],"supported":[True,True],
                    "current_visible":[True,False],"current_source_visible":[True,True],
                    "evidence_frames":[[{"frame_id":10,"timestamp_s":timestamp,"camera":"narrow"}],refs],
                    "visibility_duration_s":[0.,.1],"preview_distance_m":5.,"crop_preview_loss_m":15.}
                records.append({"accepted":True,"split":assignment.split,"family":assignment.family,
                    "group_id":assignment.group_id,"episode_id":f"{assignment.group_id}-{appearance}",
                    "label_index":100+10*decision,"timestamp_s":5.+.5*decision,"appearance":appearance,
                    "procedural":False,"direction":direction,"visibility_category":"history_crop_loss"
                    if decision < 5 else "current","action_valid":True,"qa_passed":True,"reviewed":True,
                    "visibility":visibility})
    return records,groups,assignments


def test_complete_accounting_fixture_meets_exact_quotas():
    records,groups,assignments = complete_records()
    report = audit_release(records,groups,assignments)
    assert report["qualified"],report["errors"]
    assert report["accepted_counts"] == {"train":4000,"validation":500,"test":500}
    assert report["crop_history_counts"] == {"train":1000,"validation":125,"test":125}


def test_ambiguous_records_never_inflate_accepted_counts():
    records,groups,assignments = complete_records()
    records[0] = dict(records[0],visibility_category="diagnostic_ambiguous")
    report = audit_release(records,groups,assignments)
    assert not report["qualified"]
    assert report["accepted_total"] == 4999
    assert report["diagnostic_ambiguous_count"] == 1
    assert any(error.startswith("ambiguous_record_marked_accepted") for error in report["errors"])


def test_duplicate_timestamps_and_missing_reviews_prevent_release():
    records,groups,assignments = complete_records()
    records[1] = dict(records[1],timestamp_s=records[0]["timestamp_s"],reviewed=False)
    report = audit_release(records,groups,assignments)
    assert any(error.startswith("duplicate_decision_timestamp") for error in report["errors"])
    assert any(error.startswith("unreviewed_geometry") for error in report["errors"])


def test_crop_deficit_is_not_replaced_with_easy_curves():
    records,groups,assignments = complete_records()
    records[0] = dict(records[0],visibility_category="current")
    report = audit_release(records,groups,assignments)
    assert any(error.startswith("crop_history_quota:train:999/1000") for error in report["errors"])


def test_procedural_flag_cannot_hide_procedural_geometry():
    records,groups,assignments = complete_records()
    records[0] = dict(records[0],procedural=True)
    report = audit_release(records,groups,assignments)
    assert any(error.startswith("procedural_flag_mismatch") for error in report["errors"])


def test_crop_category_string_without_pixel_ledger_cannot_fill_quota():
    records,groups,assignments = complete_records()
    records[0] = {key:value for key,value in records[0].items() if key != "visibility"}
    report = audit_release(records,groups,assignments)
    assert report["crop_history_counts"]["train"] == 999
    assert any("missing_visibility_ledger" in error for error in report["errors"])


def test_crop_evidence_future_reference_and_duration_inflation_rejected():
    records,_,_ = complete_records()
    record = records[0]
    record["visibility"]["evidence_frames"][1][0]["timestamp_s"] = record["timestamp_s"]+.05
    assert audit_crop_loss_evidence(record) == ["noncausal_or_expired_visibility_reference"]
    records,_,_ = complete_records()
    records[0]["visibility"]["visibility_duration_s"][1] = 1.0
    assert audit_crop_loss_evidence(records[0]) == ["visibility_duration_mismatch"]
