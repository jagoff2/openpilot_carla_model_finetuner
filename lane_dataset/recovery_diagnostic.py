"""Physics-only rolling recovery fixture; no image evidence or corpus labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .bootstrap import RollingWarmup,check_recording_seam
from .calibrate_extended import calibration_road
from .config import DEFAULT_ROOT, sha256_file,write_json
from .controller import BicycleMPC,MPCConfig,SteeringCalibration,VehicleState
from .controller_diagnostic import _finite_report
from .geometry import EvidenceFrame,LaneReference,wrap_angle
from .quality import qualify_episode
from .simulator import batch_control,connect,spawn_vehicle,state,world_owner


def run(root,calibration_file,port=2400,speed=2.5):
    import carla
    root=Path(root);out=root/"reports"/f"rolling-recovery-oracle-{port}-{time.time_ns()}";out.mkdir(parents=True)
    calibration=SteeringCalibration(**json.loads(Path(calibration_file).read_text()))
    if not calibration.qualified or not calibration.speed_min_mps<=speed<=calibration.speed_max_mps:
        raise ValueError("rolling recovery needs calibration covering its measured speed")
    xml=calibration_road().replace('a="30"','a="3.9"')
    (out/"straight-recovery.xodr").write_text(xml)
    client=connect(port,90);episodes=[]
    with world_owner(port,root):
        for direction in (1.,-1.):
            name="right-offset" if direction>0 else "left-offset"
            world=client.generate_opendrive_world(xml,carla.OpendriveGenerationParameters(
                vertex_distance=.5,max_road_length=50.,wall_height=0.,additional_width=.6,
                smooth_junctions=True,enable_mesh_visibility=True))
            settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
            settings.substepping=True;settings.max_substep_delta_time=.003125;settings.max_substeps=16;settings.no_rendering_mode=True
            world.apply_settings(settings)
            left=[];right=[];arc=np.arange(.01,200.,.5);map_=world.get_map()
            for s in arc:
                wp=map_.get_waypoint_xodr(1,-1,float(s));p=wp.transform.location;v=wp.transform.get_right_vector()
                center=np.array([p.x,p.y,p.z]);normal=np.array([v.x,v.y,v.z])
                left.append(center-normal*wp.lane_width/2);right.append(center+normal*wp.lane_width/2)
            reference=LaneReference(left,right,arc);ego=sensor=None;collisions=[]
            setup_rows=[];rows=[];rejections=[];started=time.perf_counter()
            try:
                ego=spawn_vehicle(world,map_.get_waypoint_xodr(1,-1,5.));bbox=ego.bounding_box
                world.tick()  # The first setup sample must include the new actor.
                sensor=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
                sensor.listen(lambda event:collisions.append(dict(frame=int(event.frame),actor=event.other_actor.type_id)))
                mpc=BicycleMPC(calibration,MPCConfig(require_calibration=False,
                    vehicle_width_m=2*bbox.extent.y,vehicle_length_m=2*bbox.extent.x))
                warmup=RollingWarmup(calibration.speed_min_mps,calibration.speed_max_mps,maximum_frames=500)
                phase="settling";integral=0.;previous_steer=0.;released_at=None;initial=None;seam=None;longitudinal_reference=speed
                for tick in range(850):
                    snapshot=world.get_snapshot();row=state(ego,snapshot)
                    s,offset=reference.project(row["position_world"][:2]);lane_yaw=float(reference.sample(s)[1])
                    yaw=row["rotation_world"][2];yaw_error=float(wrap_angle(yaw-lane_yaw))
                    lateral=-math.sin(yaw)*row["acceleration_world"][0]+math.cos(yaw)*row["acceleration_world"][1]
                    if tick>=30 and phase=="settling":phase="warmup"
                    if phase=="warmup" and warmup.observe(row["timestamp"],row["speed"],lateral,command_accepted=True,visible_reference=True):
                        phase="lateral_setup"
                    if phase=="lateral_setup" and direction*offset>=.45:
                        if abs(yaw_error-direction*math.radians(3))>math.radians(.3):
                            raise RuntimeError("physical lateral setup missed requested heading")
                        phase="recovery";released_at=row["timestamp"];mpc.reset()
                        initial=dict(offset_m=offset,heading_error_deg=math.degrees(yaw_error),speed_mps=row["speed"],route_s=s)
                        seam=check_recording_seam(setup_rows[-1],row)
                    command=None
                    if phase=="settling":throttle=0.;brake=1.;steer=0.
                    else:
                        error=speed-row["speed"];integral=float(np.clip(integral+error*.05,-2,2))
                        demand=.28*error+.08*integral+.1;throttle=max(0,demand);brake=max(0,-demand)
                        if phase=="warmup":steer=0.
                        elif phase=="lateral_setup":
                            demand_steer=float(np.clip(2.*(direction*math.radians(3)-yaw_error)-.4*row["angular_velocity_world"][2],-.10,.10))
                            steer=float(np.clip(demand_steer,previous_steer-.01,previous_steer+.01))
                        else:
                            observed=reference.admit(np.ones(len(arc),bool),
                                [(EvidenceFrame("ORACLE_DIAGNOSTIC_NOT_IMAGE",row["timestamp"],"wide"),)]*len(arc),row["timestamp"])
                            current=VehicleState(row["position_world"][0],row["position_world"][1],yaw,
                                row["speed"],row["angular_velocity_world"][2],timestamp_s=row["timestamp"],
                                lateral_acceleration_mps2=lateral)
                            command=mpc.control(current,observed,speed)
                            steer=command.carla_steer if command.accepted else 0.
                            if command.accepted:
                                longitudinal_reference=float(np.clip(longitudinal_reference+command.acceleration_mps2*.05,0,command.target_speed_mps))
                                demand=.28*(longitudinal_reference-row["speed"])+.08*integral+.1
                                throttle=max(0,demand);brake=max(0,-demand)
                            if not command.accepted:
                                throttle=0.;brake=.5;rejections.append(dict(frame=row["frame_id"],reasons=command.reasons))
                    row.update(phase=phase,route_s=s,offset_m=offset,command_steer=steer,
                        command_curvature=0. if command is None else command.curvature_1pm,
                        command_accepted=bool(command and command.accepted))
                    if phase=="recovery":rows.append(row)
                    else:setup_rows.append(row)
                    batch_control(client,ego,throttle=throttle,brake=brake,steer=steer);previous_steer=steer
                    world.tick()
                    if collisions:raise RuntimeError("rolling recovery diagnostic collision")
                    if released_at is not None and row["timestamp"]-released_at>=10.:break
                    if tick%100==0:print(json.dumps(dict(case=name,tick=tick,phase=phase,speed=row["speed"],offset_m=offset)),flush=True)
                if len(rows)<3:raise RuntimeError("rolling lateral setup did not reach recovery within bounded run")
                arrays={key:np.asarray([row[key] for row in rows]) for key in rows[0] if key!="phase"}
                setup={key:np.asarray([row[key] for row in setup_rows]) for key in setup_rows[0]}
                np.savez_compressed(out/(name+".npz"),**arrays);np.savez_compressed(out/(name+"-setup.npz"),**setup)
                yaw=arrays["rotation_world"][:,2];acc=arrays["acceleration_world"]
                lateral=-np.sin(yaw)*acc[:,0]+np.cos(yaw)*acc[:,1]
                quality=qualify_episode(timestamps_s=arrays["timestamp"],positions_xyz=arrays["position_world"],yaw_rad=yaw,
                    speeds_mps=arrays["speed"],lateral_acceleration_mps2=lateral,steering_fractions=arrays["steer"],
                    reference=reference,vehicle_length_m=2*bbox.extent.x,vehicle_width_m=2*bbox.extent.y,
                    collisions=len(collisions),rendered_boundary_agreement=None,
                    completed_maneuver=arrays["route_s"][-1]-arrays["route_s"][0]>=20.,intended_progress_m=20.,
                    recovery=True,velocity_world=arrays["velocity_world"],rotation_world=arrays["rotation_world"],
                    vehicle_height_m=2*bbox.extent.z,
                    vehicle_bbox_center_xyz=(bbox.location.x,bbox.location.y,bbox.location.z))
                report=_finite_report(dict(case=name,corpus_qualified=False,corpus_labels=0,
                    scope="oracle geometry; real rolling state created by physics controls, complete recovery interval retained; no image evidence",
                    physical_subgates_pass=quality.reasons==("rendered_boundary_agreement_missing_or_failed",) and seam["passed"],
                    quality=quality.to_dict(),initial_measured_state=initial,recording_seam=seam,
                    setup_frames=len(setup_rows),recorded_frames=len(rows),warmup=warmup.report(),
                    mpc_rejections=rejections,collisions=collisions,wall_seconds=time.perf_counter()-started,
                    frames_outside_measured_speed_domain=int(np.count_nonzero((arrays["speed"]<calibration.speed_min_mps)|(arrays["speed"]>calibration.speed_max_mps))),
                    calibration=dict(path=str(calibration_file),sha256=sha256_file(calibration_file))))
                report["mpc_prediction_jerk_limit_mps3"]=mpc.config.max_lateral_jerk_mps3
                report["vehicle_bbox"]=dict(extent=[bbox.extent.x,bbox.extent.y,bbox.extent.z],
                    center=[bbox.location.x,bbox.location.y,bbox.location.z])
                write_json(out/(name+"-report.json"),report);episodes.append(report)
                write_json(out/"report.json",dict(corpus_labels=0,corpus_qualified=False,episodes=episodes))
                print(json.dumps(report),flush=True)
            finally:
                if sensor is not None:sensor.stop();sensor.destroy()
                if ego is not None:ego.destroy()
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",default=str(DEFAULT_ROOT))
    parser.add_argument("--port",type=int,default=2400);parser.add_argument("--calibration-file",required=True)
    args=parser.parse_args();print(run(args.root,args.calibration_file,args.port),flush=True)
