"""Bounded native curb survey while the other collection GPU keeps running."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
from collections import Counter
import json
import time
import uuid

from lane_dataset.config import sha256_file,write_json
from lane_dataset.geometry import LaneReference
from lane_dataset.simulator import connect,configure_world,apply_checked_settings
from lane_behavior.curb_surface import curb_survey_owner,survey_owned_world
from lane_behavior.world_access import acquire_reservation,release_reservation,require_reserved_or_idle


def run(root,plan,port):
    root,plan=Path(root).resolve(),Path(plan).resolve()
    candidate=json.loads(plan.read_text(encoding='utf-8'))
    if candidate.get('edge_reference_selection'):
        raise ValueError('survey search must start from original outer-shoulder reference')
    task=candidate['tasks'][0];route=task['route'];road=route['unmarked_road']
    reference=LaneReference(road['road_left_edge_xyz'],road['road_right_edge_xyz'],route['arc_m'])
    if sha256_file(task['opendrive_path'])!=route['source_xodr_sha256']:
        raise ValueError('native source map hash changed')
    output=root/'reports/unmarked/surveys'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]+'.json')
    lease=acquire_reservation(root,port,duration_s=1800,purpose='independent native curb collision survey')
    try:
        deadline=time.monotonic()+1200
        while True:
            try:
                require_reserved_or_idle(root,port)
                break
            except RuntimeError as error:
                if 'still draining collection processes' not in str(error) or time.monotonic()>deadline:
                    raise
                print(json.dumps(dict(event='reserved_port_draining',port=port,detail=str(error))),flush=True)
                time.sleep(10)
        with curb_survey_owner(port,root) as ownership:
            require_reserved_or_idle(root,port)
            client=connect(port);previous=client.get_world()
            if any(a.type_id.startswith(('vehicle.','walker.','sensor.')) for a in previous.get_actors()):
                raise RuntimeError('reserved world still has actors')
            settings=previous.get_settings()
            original={k:getattr(settings,k) for k in ('synchronous_mode','fixed_delta_seconds','substepping',
                'max_substep_delta_time','max_substeps','no_rendering_mode')}
            world=None
            try:
                world,_=configure_world(client,task['town'],render=True)
                def progress(done,total,rays):
                    if done%10==0 or done==total:
                        print(json.dumps(dict(event='curb_collision_survey',done=done,total=total,rays=rays)),flush=True)
                result=survey_owned_world(world,reference,ownership=ownership,maximum_seconds=180,maximum_rays=8000,
                    progress=progress,source_bindings=dict(plan=str(plan),plan_sha256=sha256_file(plan),
                        xodr=task['opendrive_path'],xodr_sha256=route['source_xodr_sha256']))
                write_json(output,result)
                print(json.dumps(dict(event='curb_survey_finished',output=str(output),complete=result['complete'],
                    passed=result['passed'],passed_rows=sum(r['passed'] for r in result['rows']),
                    reasons=dict(Counter(r.get('reason','passed') for r in result['rows'])))),flush=True)
            finally:
                apply_checked_settings(world or previous,original)
    finally:
        release_reservation(root,lease)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--port',type=int,default=2500)
    args=parser.parse_args();run(**vars(args))
