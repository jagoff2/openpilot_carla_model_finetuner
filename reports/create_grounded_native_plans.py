"""Derive immutable on-road fixture plans with explicit capture budgets."""
import argparse
import json
from pathlib import Path

from lane_dataset.config import sha256_file,write_json
from lane_behavior.short_native import sequence_budget
from lane_behavior.sequence_guard import VERSION,ON_ROAD_VERSION,FRAME,phase_ranges


def create(source):
    source=Path(source).resolve();document=json.loads(source.read_text())
    parent_sha=sha256_file(source)
    document['parent_plan']={'path':str(source),'sha256':parent_sha}
    document['placement_revision']='fully_on_road_physically_grounded_budget_v2'
    for task in document['tasks']:
        road_id=int(task['start']['road_id']);behavior=task['unmarked_behavior']
        if road_id not in (51,52):raise ValueError('explicit surveyed native roads 51/52 only')
        parked=behavior['case']=='parked_car'
        start=(58. if parked else 55.) if road_id==51 else task['capture_start_route_m']
        actors=behavior['obstacles']
        obstacle_s=45. if road_id==52 and parked else (actors[0]['route_s_m'] if actors else
            task.get('sequence_budget',{}).get('obstacle_s_m',98.75158555002979))
        for actor in actors:
            actor['route_s_m']=obstacle_s
            actor.pop('teleport_after_spawn',None)
            actor['initial_placement_before_sensors']=True
            actor['pose_replacement_during_recording']=False
            if parked:
                actor.pop('maximum_road_intrusion_m',None)
                actor.update(placement_policy='fully_on_road_physically_settled',curb_clearance_m=.15,
                    placement='whole_measured_vehicle_inside_road_then_physical_wheel_settling',
                    required_behavior='observed_clear_obstacle_pass_and_complete_right_side_recovery')
        budget=sequence_budget(task['route']['length_m'],start_s=start,obstacle_s=obstacle_s,
            speed_mps=task['nominal_speed_mps'],ticks=900,maximum_transition_m=14. if parked else 8.)
        budget.update(version=ON_ROAD_VERSION if parked else VERSION,station_coordinate_frame=FRAME,
            physical_route_length_m=task['route']['length_m'],parent_plan_sha256=parent_sha,
            derivation='explicit physical road stations; obstacle repositioned before recording; original history/future/recovery retained')
        task.update(sequence_budget=budget,capture_start_route_m=start,capture_ticks=900,minimum_ticks=900)
        task['episode_id']=task['episode_id'].replace('-grounded-road-v1','')+'-grounded-road-v2'
        behavior.pop('station_remapping',None)
        behavior.update(controlled_scene_policy='single_static_obstacle_no_other_traffic',
            crossing_divider_allowed_only_for_observed_obstruction=parked,
            opposing_half_policy=('only_verified_visible_clear_strip_during_static_parked_pass_then_right_half_recovery_v1'
                if parked else 'whole_ego_body_stays_in_right_half_v1'),
            planned_phase_ranges_m=phase_ranges(budget,clear=not actors),
            phase_scope='explicit_on_road_budget_actual_footprints_and_recorded_timing_must_pass')
        if parked:behavior['passing_space_policy']='observed_clear_surface_during_static_parked_avoidance_only'
        else:behavior.pop('passing_space_policy',None)
    destination=source.with_name(source.name.replace('-grounded-road-v1-','-grounded-road-v2-'))
    if destination==source or destination.exists():raise FileExistsError('new immutable v2 filename required')
    write_json(destination,document)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('sources',type=Path,nargs='+')
    for source in parser.parse_args().sources:
        result=create(source);print(json.dumps(dict(plan=str(result),sha256=sha256_file(result))))
