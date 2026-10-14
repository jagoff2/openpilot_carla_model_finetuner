"""Inspect a failed recording's actual physical trajectory and boundary samples."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from lane_dataset.config import write_json
from lane_dataset.geometry import LaneReference
from lane_dataset.quality import qualify_episode

def array_items(path):
    decoder=json.JSONDecoder();buffer='';ended=False;need_read=True
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream:
        while True:
            if need_read and not ended:
                chunk=stream.read(4*1024*1024)
                buffer+=chunk;ended=not chunk;need_read=False
            buffer=buffer.lstrip(' \r\n\t[,')
            if buffer.startswith(']'):return
            try:value,last=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if ended:raise
                need_read=True
                continue
            yield value;buffer=buffer[last:]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('shard',type=Path);args=parser.parse_args();p=args.shard
    m=json.loads((p/'manifest.json').read_text());g=json.loads((p/'geometry.json').read_text())
    z=np.load(p/'partial-telemetry.npz');rejected=np.flatnonzero(~z['command_accepted']);stop=int(rejected[0]) if len(rejected) else len(z['speed'])
    boundaries=[];worst=[]
    for frame in array_items(p/'rendered-boundary-samples.json.gz'):
        i=frame['index'];samples=[dict(camera=cam,**v) for cam,data in frame['cameras'].items() for v in data['samples'] if v['measured']]
        bad=[v for v in samples if abs(v['offset_m'])>.05+1e-9]
        if bad:
            boundaries.append(dict(index=i,ego_route_s=float(z['route_s'][i]),count=len(bad),maximum=max(abs(v['offset_m']) for v in bad)))
            for v in bad:
                worst.append(dict(index=i,ego_route_s=float(z['route_s'][i]),**{k:v.get(k) for k in ('camera','side','route_s','offset_m','width_m','expected_marking_width_m','cross_section_meters_per_pixel')}))
        if i%200==0:print(json.dumps(dict(boundary_frame=i)),flush=True)
    yaw=z['rotation_world'][:stop,2];acc=z['acceleration_world'][:stop]
    d=m['vehicle_dimensions'];reference=LaneReference(g['left_xyz'],g['right_xyz'],g['arc_m'])
    result=qualify_episode(timestamps_s=z['timestamp'][:stop],positions_xyz=z['position_world'][:stop],yaw_rad=yaw,
        speeds_mps=z['speed'][:stop],lateral_acceleration_mps2=-np.sin(yaw)*acc[:,0]+np.cos(yaw)*acc[:,1],steering_fractions=z['steer'][:stop],
        reference=reference,vehicle_length_m=d['length'],vehicle_width_m=d['width'],collisions=len(m.get('collision_events',[])) if 'collision_events' in m else None,
        rendered_boundary_agreement=None,completed_maneuver=z['route_s'][stop-1]>=m['maneuver_phase_ranges']['recovery'][1],
        intended_progress_m=m['maneuver_phase_ranges']['recovery'][1]-z['route_s'][0],
        velocity_world=z['velocity_world'][:stop],rotation_world=z['rotation_world'][:stop],vehicle_height_m=d['height'],vehicle_bbox_center_xyz=d['bbox_center']).to_dict()
    report=dict(scope='Diagnostic actual recorded prefix before first rejected control; does not admit a shard or supply missing suffix frames.',
        frames=stop,first_rejected_index=stop,physical=result,boundary_failures=boundaries,
        worst_boundary_samples=sorted(worst,key=lambda v:abs(v['offset_m']),reverse=True)[:30])
    write_json(p/'recorded-turn-diagnostic.json',report)
    print(json.dumps(dict(frames=stop,physical=result,boundary_failed_frames=len(boundaries),
        boundary_failure_range=None if not boundaries else [boundaries[0],boundaries[-1]],worst=report['worst_boundary_samples'][:10])),flush=True)

if __name__=='__main__':main()
