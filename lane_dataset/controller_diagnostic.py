"""Actual CARLA physics MPC development fixtures with explicit oracle support.

No labels or camera evidence are generated. Full geometry is supplied ONLY in
this development diagnostic to isolate controller performance; these episodes
cannot pass corpus observability or rendered-boundary qualification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .config import DEFAULT_ROOT, write_json
from .controller import BicycleMPC,MPCConfig,SteeringCalibration,VehicleState
from .geometry import EvidenceFrame,LaneReference
from .procedural import ProceduralParameters,build_opendrive,curve_segments
from .quality import qualify_episode
from .simulator import batch_control,connect,spawn_vehicle,state,world_owner


def _finite_report(value):
    if isinstance(value,dict):return {key:_finite_report(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_finite_report(item) for item in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):return None
    return value


def run(root,calibration_file,port=2400,families=("constant_radius","tightening","s_bend","hairpin"),
        speed_mps=5.,radius_m=22.,turn="left",offset_m=0.,heading_error_deg=0.):
    import carla
    root=Path(root);out=root/"reports"/f"controller-oracle-diagnostic-{port}-{time.time_ns()}";out.mkdir(parents=True)
    calibration_document=json.loads(Path(calibration_file).read_text())
    calibrations=calibration_document.get("calibrations_by_speed",{str(float(speed_mps)):calibration_document})
    target=float(speed_mps);calibration=SteeringCalibration(**calibrations[str(target)])
    if not calibration.qualified or not calibration.speed_min_mps<=target<=calibration.speed_max_mps:
        raise ValueError("diagnostic target speed needs a qualified measured calibration")
    recovery=abs(offset_m)>0 or abs(heading_error_deg)>0
    client=connect(port,90);reports=[]
    with world_owner(port,root):
        for family in families:
            params=ProceduralParameters(family,radius_m,turn=turn,lane_width_m=3.9,approach_m=40.,exit_m=55.,
                                        bank_rad=.04 if family=="s_bend" else 0.,split="development")
            xml=build_opendrive(params);(out/(family+".xodr")).write_text(xml)
            world=client.generate_opendrive_world(xml,carla.OpendriveGenerationParameters(
                vertex_distance=.5,max_road_length=50.,wall_height=0.,additional_width=.6,
                smooth_junctions=True,enable_mesh_visibility=True))
            settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
            settings.substepping=True;settings.max_substep_delta_time=.003125;settings.max_substeps=16;settings.no_rendering_mode=True
            world.apply_settings(settings)
            length=sum(segment.length_m for segment in curve_segments(params));map_=world.get_map()
            left=[];right=[];arc=[]
            for s in np.arange(.01,length-.01,.25):
                wp=map_.get_waypoint_xodr(1,-1,float(s))
                if wp is None:continue
                transform=wp.transform;v=transform.get_right_vector();p=transform.location
                center=np.array([p.x,p.y,p.z]);normal=np.array([v.x,v.y,v.z])
                left.append(center-normal*wp.lane_width/2);right.append(center+normal*wp.lane_width/2);arc.append(s)
            reference=LaneReference(left,right,arc);ego=sensor=None;collisions=[];rows=[];commands=[];rejections=[]
            mpc=BicycleMPC(calibration,MPCConfig(require_calibration=False))
            started=time.perf_counter()
            try:
                ego=spawn_vehicle(world,map_.get_waypoint_xodr(1,-1,5.),offset_m,heading_error_deg)
                bbox=ego.bounding_box
                mpc=BicycleMPC(calibration,MPCConfig(require_calibration=False,
                    vehicle_width_m=2*bbox.extent.y,vehicle_length_m=2*bbox.extent.x))
                sensor=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
                sensor.listen(lambda event:collisions.append({"frame":int(event.frame),"actor":event.other_actor.type_id}))
                integral=0.;last_s=5.;finished=False
                maximum_ticks=int(math.ceil((length-25)/target/.05))+240
                for tick in range(maximum_ticks):
                    snapshot=world.get_snapshot();row=state(ego,snapshot)
                    # Admission is deliberately oracle-tagged and never exported
                    # to training. This isolates physical controller performance.
                    observed=reference.admit(np.ones(len(arc),bool),
                        [(EvidenceFrame("ORACLE_DIAGNOSTIC_NOT_IMAGE",row["timestamp"],"wide"),)]*len(arc),row["timestamp"])
                    current=VehicleState(row["position_world"][0],row["position_world"][1],row["rotation_world"][2],
                        row["speed"],row["angular_velocity_world"][2],timestamp_s=row["timestamp"],
                        lateral_acceleration_mps2=-math.sin(row["rotation_world"][2])*row["acceleration_world"][0]+
                            math.cos(row["rotation_world"][2])*row["acceleration_world"][1])
                    if tick<30:
                        batch_control(client,ego,brake=1.);command=None
                    else:
                        command=mpc.control(current,observed,target)
                        if command.accepted:
                            error=command.target_speed_mps-row["speed"];integral=float(np.clip(integral+error*.05,-2,2))
                            demand=.28*error+.08*integral+.1
                            batch_control(client,ego,throttle=max(0,demand),brake=max(0,-demand),steer=command.carla_steer)
                        else:
                            rejections.append({"tick":tick,"reasons":command.reasons,"margin":command.min_constraint_margin})
                            batch_control(client,ego,brake=.5)
                    world.tick()
                    # Recovery starts at the first control after settling. Its
                    # difficult early interval is retained, including raw launch
                    # dynamics; it must never be hidden inside a warmup window.
                    if tick>=(30 if recovery else 160):
                        rows.append(row)
                        commands.append(0. if command is None else command.curvature_1pm)
                        last_s,_=reference.project(row["position_world"][:2],last_s)
                        if last_s>=length-30.:
                            finished=True;break
                    if collisions:break
                    if tick%100==0:print(f"{family} tick{tick} speed{row['speed']:.2f} progress{last_s:.1f} rejects{len(rejections)}",flush=True)
                arrays={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
                arrays["command_curvature"]=np.asarray(commands);np.savez_compressed(out/(family+".npz"),**arrays)
                yaw=arrays["rotation_world"][:,2];acc=arrays["acceleration_world"]
                lateral=-np.sin(yaw)*acc[:,0]+np.cos(yaw)*acc[:,1]
                quality=qualify_episode(timestamps_s=arrays["timestamp"],positions_xyz=arrays["position_world"],yaw_rad=yaw,
                    speeds_mps=arrays["speed"],lateral_acceleration_mps2=lateral,steering_fractions=arrays["steer"],
                    reference=reference,vehicle_length_m=2*bbox.extent.x,vehicle_width_m=2*bbox.extent.y,
                    collisions=len(collisions),rendered_boundary_agreement=None,completed_maneuver=finished,
                    intended_progress_m=max(0.,length-31.-reference.project(arrays["position_world"][0])[0]),
                    recovery=recovery,velocity_world=arrays["velocity_world"],rotation_world=arrays["rotation_world"],
                    vehicle_height_m=2*bbox.extent.z,
                    vehicle_bbox_center_xyz=(bbox.location.x,bbox.location.y,bbox.location.z))
                item={"family":family,"corpus_qualified":False,"admission":"oracle diagnostic only, no image support",
                    "physical_subgates_pass":quality.reasons==("rendered_boundary_agreement_missing_or_failed",),
                    "quality":quality.to_dict(),"mpc_rejections":rejections,"collisions":collisions,
                    "wall_seconds":time.perf_counter()-started,"captured_frames":len(rows),"bank_rad":params.bank_rad,
                    "turn":turn,"radius_m":radius_m,"target_speed_mps":target,"recovery":recovery,
                    "initial_offset_m":offset_m,"initial_heading_error_deg":heading_error_deg,
                    "calibration_file":str(calibration_file),"frames_outside_measured_speed_domain":int(np.count_nonzero(
                        (arrays["speed"]<calibration.speed_min_mps)|(arrays["speed"]>calibration.speed_max_mps))),
                    "initialization_scope":"recovery launch retained from first control" if recovery else "centered physical startup precedes recorded diagnostic"}
                item=_finite_report(item)
                reports.append(item);write_json(out/(family+"-report.json"),item)
                write_json(out/"report.json",{"status":"development_diagnostic","corpus_labels":0,"episodes":reports})
                print(json.dumps({"family":family,"quality":quality.to_dict()}),flush=True)
            finally:
                if sensor is not None:sensor.stop();sensor.destroy()
                if ego is not None:ego.destroy()
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",default=str(DEFAULT_ROOT))
    parser.add_argument("--port",type=int,default=2400);parser.add_argument("--calibration-file",required=True)
    parser.add_argument("--families",nargs="+",default=["constant_radius","tightening","s_bend","hairpin"])
    parser.add_argument("--speed",type=float,default=5.);parser.add_argument("--radius",type=float,default=22.)
    parser.add_argument("--turn",choices=("left","right"),default="left")
    parser.add_argument("--offset",type=float,default=0.);parser.add_argument("--heading",type=float,default=0.)
    args=parser.parse_args();print(run(args.root,args.calibration_file,args.port,args.families,args.speed,args.radius,
        args.turn,args.offset,args.heading),flush=True)
