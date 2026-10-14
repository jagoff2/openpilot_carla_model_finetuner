"""Bounded physics-only extended steering calibration on an empty wide road.

Uses only numpy/CARLA at capture time. Run ``fit`` using a scipy environment.
These maneuvers identify an actuator model; the wide road is not corpus data.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np

from .config import DEFAULT_ROOT, write_json
from .simulator import batch_control, connect, spawn_vehicle, state, world_owner


def calibration_road():
    return '''<?xml version="1.0"?><OpenDRIVE>
<header revMajor="1" revMinor="4" name="development-steering-calibration" version="1.0" north="0" south="0" east="0" west="0"/>
<road name="empty-calibration-road" length="600" id="1" junction="-1"><link/>
<planView><geometry s="0" x="0" y="0" hdg="0" length="600"><line/></geometry></planView>
<elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
<lateralProfile/><lanes><laneSection s="0">
<left><lane id="1" type="driving" level="false"><link/><width sOffset="0" a="30" b="0" c="0" d="0"/>
<roadMark sOffset="0" type="solid" color="white" width="0.15"/></lane></left>
<center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="solid" color="yellow" width="0.15"/></lane></center>
<right><lane id="-1" type="driving" level="false"><link/><width sOffset="0" a="30" b="0" c="0" d="0"/>
<roadMark sOffset="0" type="solid" color="white" width="0.15"/></lane></right>
</laneSection></lanes><objects/><signals/></road></OpenDRIVE>'''


def pulse_schedule(repetitions=3, amplitudes=(.05,.1,.2,.25)):
    # Each +,-,-,+ lobe set cancels its first-order heading/displacement change.
    if (not amplitudes or any(not math.isfinite(value) or not 0<value<=1 for value in amplitudes)
        or not isinstance(repetitions,int) or repetitions<1):
        raise ValueError("calibration requires positive repetitions and steering fractions in (0,1]")
    pattern=[]
    for _ in range(repetitions):
        for amplitude in amplitudes:
            for value in (0.,amplitude,0.,-amplitude,0.,-amplitude,0.,amplitude,0.):
                pattern.extend([value]*10)
    return np.asarray(pattern,float)


def capture(root,port=2400,speeds=(3.,5.),amplitudes=(.05,.1,.2,.25)):
    import carla
    speeds=tuple(float(speed) for speed in speeds)
    if not speeds or len(set(speeds))!=len(speeds) or any(not math.isfinite(speed) or speed<=0 for speed in speeds):
        raise ValueError("calibration speeds must be distinct, finite and positive")
    pulse=pulse_schedule(amplitudes=amplitudes)
    root=Path(root);out=root/"reports"/f"calibration-extended-{port}-{time.time_ns()}";out.mkdir(parents=True)
    client=connect(port,timeout=90);collision_events=[]
    with world_owner(port,root):
        previous=client.get_world();settings=previous.get_settings()
        settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.substepping=True
        settings.max_substeps=16;settings.max_substep_delta_time=.003125;settings.no_rendering_mode=True
        previous.apply_settings(settings)
        xml=calibration_road();(out/"calibration.xodr").write_text(xml)
        world=client.generate_opendrive_world(xml,carla.OpendriveGenerationParameters(
            vertex_distance=2.,max_road_length=50.,wall_height=0.,additional_width=.6,
            smooth_junctions=True,enable_mesh_visibility=True))
        world.apply_settings(settings)
        got=world.get_settings();keys=("synchronous_mode","fixed_delta_seconds","substepping",
            "max_substeps","max_substep_delta_time","no_rendering_mode")
        readback={key:getattr(got,key) for key in keys}
        if not got.synchronous_mode or not math.isclose(got.fixed_delta_seconds,.05,abs_tol=1e-7):
            raise RuntimeError("synchronization settings readback failed")
        metadata={"status":"capturing","port":port,"settings":readback,"purpose":"development actuator calibration only",
                  "server_version":client.get_server_version(),"captures":[],"road_departure":False,
                  "steering_pulse_amplitudes":list(amplitudes)}
        write_json(out/"manifest.json",metadata)
        for target_speed in speeds:
            ego=sensor=None
            try:
                wp=world.get_map().get_waypoint_xodr(1,-1,50.)
                if wp is None: raise RuntimeError("generated calibration lane missing")
                ego=spawn_vehicle(world,wp)
                physics=ego.get_physics_control()
                metadata.update(wheelbase_m=abs(physics.wheels[0].position.x-physics.wheels[2].position.x)/100.,
                    wheel_steer_degrees=[float(w.max_steer_angle) for w in physics.wheels],vehicle=ego.type_id,
                    steering_curve=[[float(p.x),float(p.y)] for p in physics.steering_curve])
                sensor=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
                sensor.listen(lambda event:collision_events.append({"frame":int(event.frame),"other_actor":str(event.other_actor.type_id)}))
                rows=[];integral=0.;warmup=160
                for tick in range(warmup+len(pulse)):
                    current=float(math.hypot(ego.get_velocity().x,ego.get_velocity().y))
                    desired=target_speed if tick>=30 else 0.
                    error=desired-current;integral=float(np.clip(integral+error*.05,-2,2))
                    demand=.28*error+.08*integral+(.1 if desired else 0.)
                    steer=float(pulse[tick-warmup]) if tick>=warmup else 0.
                    batch_control(client,ego,throttle=max(0,demand),brake=max(0,-demand),steer=steer)
                    frame=world.tick();snapshot=world.get_snapshot()
                    if snapshot.frame!=frame:raise RuntimeError("snapshot frame mismatch")
                    row=state(ego,snapshot);row["command_steer"]=steer;rows.append(row)
                    if collision_events: raise RuntimeError("calibration collision")
                    if not (2.5<row["position_world"][1]<27.5):
                        metadata["road_departure"]=True;raise RuntimeError("calibration left wide lane body clearance")
                arrays={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
                filename=f"speed-{target_speed:g}-samples.npz";np.savez_compressed(out/filename,**arrays)
                metadata["captures"].append({"target_speed_mps":target_speed,"file":filename,
                    "warmup_frames":warmup,"frames":len(rows),"speed_min_mps":float(arrays["speed"][warmup:].min()),
                    "speed_max_mps":float(arrays["speed"][warmup:].max()),
                    "lateral_y_min_m":float(arrays["position_world"][:,1].min()),
                    "lateral_y_max_m":float(arrays["position_world"][:,1].max())})
                write_json(out/"manifest.json",metadata)
                print(f"captured {target_speed:g} m/s, {len(rows)} frames",flush=True)
            finally:
                if sensor is not None:sensor.stop();sensor.destroy()
                if ego is not None:ego.destroy()
        metadata.update(status="captured_not_fitted",collision_events=collision_events)
        write_json(out/"manifest.json",metadata)
    return out


def fit(directory):
    from .controller import fit_steering_calibration, fit_rate_limited_calibration
    directory=Path(directory);metadata=json.loads((directory/"manifest.json").read_text());fits={};baselines={}
    for capture_info in metadata["captures"]:
        raw=np.load(directory/capture_info["file"]);start=capture_info["warmup_frames"]
        result=fit_rate_limited_calibration(raw["timestamp"][start:-1],raw["command_steer"][start+1:],
            raw["speed"][start:-1],raw["angular_velocity_world"][start:-1,2],metadata["wheelbase_m"],
            math.radians(max(metadata["wheel_steer_degrees"])))
        # CARLA reports the actor reference position, not the bicycle rear axle.
        # Fit its body-lateral velocity / yaw-rate relation using only training
        # frames; independently check the final chronological hold-out segment.
        yaw=raw["rotation_world"][start:-1,2];velocity=raw["velocity_world"][start:-1]
        yaw_rate=raw["angular_velocity_world"][start:-1,2]
        lateral_velocity=-np.sin(yaw)*velocity[:,0]+np.cos(yaw)*velocity[:,1]
        cut=int(len(yaw)*.7);active=np.abs(yaw_rate)>.05
        train=active&(np.arange(len(yaw))<cut);holdout=active&(np.arange(len(yaw))>=cut)
        offset=float(np.sum(lateral_velocity[train]*yaw_rate[train])/np.sum(yaw_rate[train]**2))
        residual=lateral_velocity[holdout]-offset*yaw_rate[holdout]
        reference_rmse=float(np.sqrt(np.mean(residual**2)))
        result=replace(result,reference_offset_m=offset,qualified=result.qualified and reference_rmse<=.03 and 0.<offset<metadata["wheelbase_m"])
        fits[str(capture_info["target_speed_mps"])]=result.to_dict()
        artifact=directory/f"fit-speed-{capture_info['target_speed_mps']:g}.json"
        if artifact.exists() and json.loads(artifact.read_text())!=result.to_dict():
            raise RuntimeError("refusing to overwrite a different measured calibration artifact")
        if not artifact.exists():write_json(artifact,result.to_dict())
        capture_info["reference_offset_holdout_velocity_rmse_mps"]=reference_rmse
        baseline=fit_steering_calibration(raw["timestamp"][start:-1],raw["command_steer"][start+1:],
            raw["speed"][start:-1],raw["angular_velocity_world"][start:-1,2],metadata["wheelbase_m"],
            math.radians(max(metadata["wheel_steer_degrees"])))
        baselines[str(capture_info["target_speed_mps"])]=baseline.to_dict()
    report={"calibrations_by_speed":fits,"reference_offset_validation":[{
        "target_speed_mps":row["target_speed_mps"],"heldout_lateral_velocity_rmse_mps":row["reference_offset_holdout_velocity_rmse_mps"]}
        for row in metadata["captures"]],"all_heldout_fits_pass":bool(fits) and all(v["qualified"] for v in fits.values()),
        "physical_episode_quality_qualified":False,
        "scope":"Steering calibration only; no lane-centering, visibility or corpus qualification inferred."}
    write_json(directory/"fit-report.json",report)
    write_json(directory/"linear-model-rejection-report.json",{"calibrations_by_speed":baselines,
        "reason":"First-order response misses slew-limited transitions; thresholds unchanged for corrected model."})
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("capture","fit"))
    parser.add_argument("--root",default=str(DEFAULT_ROOT));parser.add_argument("--port",type=int,default=2400)
    parser.add_argument("--directory");parser.add_argument("--speeds",type=float,nargs="+",default=[3.,5.])
    parser.add_argument("--amplitudes",type=float,nargs="+",default=[.05,.1,.2,.25]);args=parser.parse_args()
    print(capture(args.root,args.port,args.speeds,args.amplitudes) if args.mode=="capture" else json.dumps(fit(args.directory),indent=2),flush=True)
