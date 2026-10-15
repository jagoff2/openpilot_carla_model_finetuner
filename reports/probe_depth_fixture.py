"""Compare an updated depth fixture against live sensors on failed spawn poses."""
import argparse
import importlib.util
import json
from pathlib import Path

from lane_dataset.capture import _depth_fixture, _finite_json
from lane_dataset.config import write_json
from lane_dataset.geometry import LaneReference
from lane_dataset.preprocessing import StockPreprocessor
from lane_dataset.simulator import (connect, configure_world, world_owner,
    spawn_vehicle, CameraRig, batch_control)
from lane_dataset.capture import OPTICAL_FROM_CARLA


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--candidate-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('jobs',nargs='+')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('lane_dataset.depth_fixture_candidate',args.candidate_source)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    reports=[]
    with world_owner(args.port,args.root):
        for job in args.jobs:
            output=args.root/'reports/automatic'
            result=json.loads((output/(job+'.job.result.json')).read_text())
            shard=Path(result['shard'])
            task=json.loads((shard/'requested-task.json').read_text())
            geometry=json.loads((shard/'geometry.json').read_text())
            reference=LaneReference(geometry['left_xyz'],geometry['right_xyz'],geometry['arc_m'])
            client=connect(args.port);world,_=configure_world(client,task['town'],render=True)
            waypoint=world.get_map().get_waypoint_xodr(**task['start'])
            ego=spawn_vehicle(world,waypoint,task.get('offset_m',0),task.get('heading_error_deg',0));rig=None
            try:
                batch_control(client,ego,brake=1.)
                for _ in range(30):world.tick()
                rig=CameraRig(world,ego,semantic_qa=True)
                for _ in range(3):
                    world.tick();images=rig.read(world.get_snapshot())
                prep=StockPreprocessor()
                before=_depth_fixture(images,prep,reference,world)
                after=candidate.collision_depth_fixture(images,prep,reference,world,OPTICAL_FROM_CARLA)
                row=dict(job=job,town=task['town'],before=before,after=after,
                    passed=all(v['qualified'] for v in after.values()))
                reports.append(row)
                print(json.dumps(dict(job=job,town=task['town'],passed=row['passed'],
                    before={k:{a:v[a] for a in ('qualified','reason')} for k,v in before.items()},
                    after={k:{a:v[a] for a in ('qualified','reason','maximum_residual_m') if a in v} for k,v in after.items()})),flush=True)
            finally:
                if rig:rig.close()
                ego.destroy()
    write_json(args.output,_finite_json(dict(passed=all(r['passed'] for r in reports),
        scope='live sensor depth comparison at failed spawn poses; not corpus admission',reports=reports)))
    return 0 if all(r['passed'] for r in reports) else 2


if __name__=='__main__':raise SystemExit(main())
