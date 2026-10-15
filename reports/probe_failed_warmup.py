"""Replay saved poses for sensor diagnosis only; no training data is produced."""
import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from PIL import Image
import psutil

from lane_dataset.capture import OPTICAL_FROM_CARLA, _finite_json
from lane_dataset.config import write_json
from lane_dataset.depth_sampling import surface_normals
from lane_dataset.geometry import LaneReference
from lane_dataset.preprocessing import StockPreprocessor, image_to_nv12, split_nv12
from lane_dataset.simulator import world_owner, connect, configure_world, spawn_vehicle, CameraRig
from lane_dataset.visibility import CameraObservation, probe_lane_evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--shard',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--supervisor',type=int,required=True)
    args=parser.parse_args()
    task=json.loads((args.shard/'requested-task.json').read_text())
    geometry=json.loads((args.shard/'geometry.json').read_text())
    fixture=json.loads((args.shard/'depth-fixture.json').read_text())
    telemetry=dict(np.load(args.shard/'bootstrap-telemetry.npz',allow_pickle=False))
    output=args.root/'reports'/('warmup-sensor-diagnosis-'+task['episode_id'])
    output.mkdir(exist_ok=True)
    supervisor=psutil.Process(args.supervisor)
    command=supervisor.cmdline()
    if 'reports.automatic_corpus' not in command or '--job' in command or str(args.root) not in command:
        raise ValueError('unexpected supervisor identity')
    # Prevent dispatch onto this port during the short diagnostic. Existing
    # captures on the other GPU continue independently. Always resume ownership.
    deadline=time.monotonic()+1200
    print(json.dumps(dict(waiting_for_capture_port=args.port)),flush=True)
    while True:
        supervisor.suspend()
        ownership=ExitStack()
        try:
            ownership.enter_context(world_owner(args.port,args.root))
            break
        except PermissionError:
            ownership.close();supervisor.resume()
            if time.monotonic()>deadline:raise TimeoutError('capture port did not become free')
            time.sleep(2)
        except BaseException:
            ownership.close();supervisor.resume();raise
    try:
        with ownership:
            import carla
            client=connect(args.port);world,_=configure_world(client,task['town'],render=True)
            start=task['route']['start']
            ego=spawn_vehicle(world,world.get_map().get_waypoint_xodr(**start));rig=None
            try:
                ego.set_simulate_physics(False)
                rig=CameraRig(world,ego,semantic_qa=True)
                prep=StockPreprocessor()
                reference=LaneReference(geometry['left_xyz'],geometry['right_xyz'],geometry['arc_m'])
                n=len(reference.s);points=np.r_[reference.left_xyz,reference.right_xyz]
                ids=tuple([f'L{i}' for i in range(n)]+[f'R{i}' for i in range(n)])
                arc=np.tile(reference.s,2);normals=surface_normals(reference)
                reports=[]
                for index in (0,200,len(telemetry['frame_id'])-1):
                    x,y,z=telemetry['position_world'][index]
                    roll,pitch,yaw=np.rad2deg(telemetry['rotation_world'][index])
                    ego.set_transform(carla.Transform(carla.Location(x=float(x),y=float(y),z=float(z)),
                        carla.Rotation(roll=float(roll),pitch=float(pitch),yaw=float(yaw))))
                    for _ in range(12):
                        world.tick();snapshot=world.get_snapshot();images=rig.read(snapshot)
                    summary=dict(index=index,pose=telemetry['position_world'][index].tolist(),cameras={})
                    for cam,profile,warp in zip(('narrow','wide'),prep.profiles,prep.warps):
                        nv12=image_to_nv12(images[cam+'_rgb'],'RGB');planes=split_nv12(*nv12)
                        tags=images[cam+'_semantic_segmentation']
                        observation=CameraObservation(snapshot.frame,snapshot.timestamp.elapsed_seconds,cam,
                            images[cam+'_rgb'],images[cam+'_depth'],OPTICAL_FROM_CARLA@images[cam+'_rgb_camera_from_world'],
                            profile.K,warp,depth_mode=fixture[cam]['depth_mode'],depth_validated=True,
                            luma=planes[0],chroma_u=planes[1],chroma_v=planes[2],surface_normals_world=normals,
                            depth_pixel_center_offset=.5,marking_mask=tags==int(carla.CityObjectLabel.RoadLines),
                            road_mask=np.isin(tags,[int(carla.CityObjectLabel.Roads),int(carla.CityObjectLabel.RoadLines)]),
                            road_exterior_mask=np.isin(tags,[int(carla.CityObjectLabel.Sidewalks),int(carla.CityObjectLabel.Terrain),int(carla.CityObjectLabel.Ground)]))
                        evidence=probe_lane_evidence(observation,points,ids,arc)
                        summary['cameras'][cam]=dict(source_left=int(evidence.source_visible[:n].sum()),
                            source_right=int(evidence.source_visible[n:].sum()),crop_left=int(evidence.crop_visible[:n].sum()),
                            crop_right=int(evidence.crop_visible[n:].sum()))
                        Image.fromarray(images[cam+'_rgb']).save(output/f'{index}-{cam}.png')
                        np.savez_compressed(output/f'{index}-{cam}.npz',rgb=images[cam+'_rgb'],depth=images[cam+'_depth'],
                            semantic=tags,camera_from_world=observation.camera_from_world)
                        write_json(output/f'{index}-{cam}-evidence.json',_finite_json(evidence.compact()))
                    reports.append(summary);print(json.dumps(summary),flush=True)
                write_json(output/'summary.json',dict(scope='diagnostic pose replay, not corpus data',source=str(args.shard),rows=reports))
            finally:
                if rig:rig.close()
                ego.destroy()
    finally:
        supervisor.resume()
    print(json.dumps(dict(output=str(output),supervisor_resumed=args.supervisor)),flush=True)


if __name__=='__main__':main()
