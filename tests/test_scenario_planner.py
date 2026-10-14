import pytest

from lane_dataset.scenario_planner import (PlannerConfig, candidate_conflicts,
    choose_independent,direct_junction_links,_procedural_parameters,optimize_native_selection,
    recording_ticks)
from lane_dataset.splits import FAMILIES,GeometrySection,RoadInterval,group_geometry_sections


def make_section(name,start,end,family,road="1",town="Town03",**kwargs):
    return GeometrySection(name,town,family,(RoadInterval(road,start,end),),**kwargs)


def test_unselected_bridging_route_does_not_collapse_independent_geometry():
    sections=[make_section("a",0,20,"straight_recovery"),
              make_section("bridge",15,85,"intersection_turn"),
              make_section("c",80,100,"constant_radius")]
    assert len(group_geometry_sections(sections))==1
    graph=candidate_conflicts(sections)
    targets={family:0 for family in FAMILIES}
    targets.update(straight_recovery=1,constant_radius=1)
    chosen=choose_independent(sections,graph,targets,attempts=3)
    assert chosen==[2,0]
    assert len(group_geometry_sections([sections[i] for i in chosen]))==2


def test_shared_junction_and_reverse_intervals_still_conflict():
    sections=[make_section("a",0,20,"straight_recovery",junction_ids=("1",)),
              make_section("b",10,5,"straight_recovery"),
              make_section("c",0,20,"constant_radius",road="2",junction_ids=("1",))]
    graph=candidate_conflicts(sections)
    assert graph[0]=={1,2}
    chosen=choose_independent(sections,graph,{family:3 for family in FAMILIES},attempts=3)
    assert all(other not in graph[index] for i,index in enumerate(chosen) for other in chosen[i+1:])


def test_direct_junction_connections_exclude_arbitrary_city_reachability():
    xml='''<OpenDRIVE>
    <road id="1" junction="10"><link><successor elementType="road" elementId="2"/></link></road>
    <road id="2" junction="11"/>
    <road id="3" junction="-1"><link><predecessor elementType="junction" elementId="11"/>
    <successor elementType="junction" elementId="12"/></link></road>
    <junction id="13"><connection connectingRoad="4"/></junction>
    <junction id="14"><connection connectingRoad="4"/></junction></OpenDRIVE>'''
    assert direct_junction_links(xml)==(("10","11"),("13","14"))


def test_route_envelope_includes_temporal_and_preview_budget():
    PlannerConfig().validate()
    with pytest.raises(ValueError,match="envelope"):
        PlannerConfig(route_envelope_m=70.).validate()
    with pytest.raises(ValueError,match="envelope"):
        PlannerConfig(route_envelope_m=90.,warmup_distance_budget_m=10.).validate()
    PlannerConfig(route_envelope_m=100.,warmup_distance_budget_m=10.).validate()


def test_approved_expansion_keeps_test_map_required():
    config=PlannerConfig()
    assert {"Town01","Town02","Town06","Town07"}.issubset(config.towns)
    for town in config.towns:
        make_section(town,0,20,"straight_recovery",town=town)
    for towns in (("Town01",), ("Town05","Town99"), ("Town05","Town05")):
        with pytest.raises(ValueError,match="test-only Town05"):
            PlannerConfig(towns=towns).validate()


def test_recording_does_not_consume_warmup_or_preview_distance():
    config=PlannerConfig(route_envelope_m=200.,minimum_episode_s=40.,
                         warmup_distance_budget_m=55.,planned_preview_m=25.)
    config.validate()
    assert recording_ticks(200.,config)==800
    assert recording_ticks(200.,config,procedural=True)==960
    assert recording_ticks(200.,config,procedural=True,capture_start_m=10.)==880
    for procedural in (False,True):
        ticks=recording_ticks(200.,config,procedural=procedural)
        assert ticks*.05*config.nominal_speed_mps+55.+25.<=200.
    with pytest.raises(ValueError,match="cannot contain"):
        recording_ticks(179.,config)


def test_all_procedural_split_radius_lattices_are_disjoint():
    values={split:{_procedural_parameters(split,"severe_hairpin",index).radius_m for index in range(20)}
            for split in ("train","validation","test")}
    assert not values["train"]&values["validation"]
    assert not values["train"]&values["test"]
    assert not values["test"]&values["validation"]
    assert len({_procedural_parameters("train","severe_hairpin",index).geometry_group for index in range(20)})==20


def test_binary_selection_preserves_native_only_and_overlap_constraints():
    sections=[make_section("straight",0,20,"straight_recovery"),
              make_section("turn",0,20,"intersection_turn",road="2"),
              make_section("curve",50,70,"constant_radius"),
              make_section("conflict",55,65,"severe_hairpin")]
    graph=candidate_conflicts(sections)
    targets={family:1 for family in FAMILIES}
    chosen,report=optimize_native_selection(sections,graph,targets,[0,1,2],time_limit_s=1.)
    assert len(chosen)==3
    assert {0,1}.issubset(chosen)
    assert report["proven_optimal"]
    assert not ({2,3}.issubset(chosen))


def test_binary_selection_does_not_satisfy_turn_quotas_with_one_direction():
    sections=[make_section(f"s{i}",0,20,"straight_recovery",road=f"s{i}") for i in range(3)]
    sections.extend(make_section(f"l{i}",0,20,"intersection_turn",road=f"l{i}",direction="left") for i in range(4))
    sections.append(make_section("r",0,20,"intersection_turn",road="r",direction="right"))
    targets={family:0 for family in FAMILIES}
    targets.update(straight_recovery=3,intersection_turn=3)
    chosen,_=optimize_native_selection(sections,candidate_conflicts(sections),targets,[],time_limit_s=1.)
    assert len(chosen)==6
    assert sum(sections[index].direction=="left" for index in chosen)==2
    assert sum(sections[index].direction=="right" for index in chosen)==1
