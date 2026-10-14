"""Physics-driven development recorder and strictly gated resumable generation."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
import traceback
import shutil
import gzip

import numpy as np

from .config import Config, PROJECT, sha256_file, write_json
from .controller import BicycleMPC, MPCConfig, SteeringCalibration, VehicleState
from .bootstrap import RollingWarmup, check_recording_seam
from .observed_corridor import provenance as observable_provenance
from .terminal_suffix import validate_terminal_stop, suffix_stop_control
from .geometry import LaneReference, ObservableReference, MAX_OBSERVED_KNOT_GAP_M
from .inventory import sample_route, route_document
from .maneuvers import route_desire
from .rendered_boundaries import measure_camera, summarize_frames, bind_marking_widths
from .road_surface import survey_road_surface
from .depth_sampling import surface_normals
from .initialization import PhysicalInitializer
from .preprocessing import StockPreprocessor, reference_module, image_to_nv12, split_nv12
from .quality import qualify_episode
from .simulator import CameraRig, batch_control, configure_world, connect, spawn_vehicle, state, world_owner, apply_checked_settings
from .storage import Store
from .traffic import SurroundingTraffic, TrafficConfig
from .visibility import CameraObservation, EvidenceHistory, probe_lane_evidence, project_world_points, validate_depth_geometry

OPTICAL_FROM_CARLA = np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],float)


def _finite_json(value):
    if isinstance(value,dict): return {k:_finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_finite_json(v) for v in value]
    if isinstance(value,np.ndarray): return _finite_json(value.tolist())
    if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value,np.generic): return value.item()
    return value


def _calibration(root, nominal_speed_mps=5.):
    if not math.isfinite(nominal_speed_mps) or nominal_speed_mps<=0:
        raise ValueError("capture nominal speed must be finite and positive")
    candidates=[]
    for source in (Path(root)/"reports").glob("calibration-*/fit*.json"):
        document=json.loads(source.read_text())
        if "calibrations_by_speed" in document:continue
        fitted=SteeringCalibration(**document)
        if fitted.qualified and fitted.speed_min_mps<=nominal_speed_mps<=fitted.speed_max_mps:
            distance=abs((fitted.speed_min_mps+fitted.speed_max_mps)/2-nominal_speed_mps)
            candidates.append((distance,-source.stat().st_mtime,str(source),source))
    if not candidates:
        raise RuntimeError(f"capture requires qualified measured steering calibration covering {nominal_speed_mps:g} m/s")
    source=min(candidates)[-1];digest=sha256_file(source)
    frozen=Path(root)/"calibrations"/digest
    deadline=time.monotonic()+30.
    while not frozen.exists():
        lock=world_owner("calibration-"+digest,root)
        try:
            lock.__enter__()
        except OSError:
            if time.monotonic()>=deadline:
                raise TimeoutError("timed out waiting for calibration snapshot writer")
            time.sleep(.05)
            continue
        try:
            if not frozen.exists():
                frozen.parent.mkdir(parents=True,exist_ok=True)
                temporary=frozen.with_name(digest+".partial-"+str(time.time_ns()))
                shutil.copytree(source.parent,temporary)
                if sha256_file(temporary/source.name)!=digest:
                    raise RuntimeError("calibration changed while creating immutable snapshot")
                temporary.rename(frozen)
        finally:
            lock.__exit__(None,None,None)
    path=frozen/source.name
    if sha256_file(path)!=digest:
        raise RuntimeError("calibration changed while creating immutable snapshot")
    return SteeringCalibration(**json.loads(path.read_text())),dict(path=str(path),sha256=digest)


def _depth_fixture(images, preprocessor, reference):
    """Road-surface samples use independent OpenDRIVE 3D boundary coordinates.

    A per-camera unique forward-vs-radial fit is required; depth is never
    credited merely because it looks plausible. Samples and transforms are
    retained for review. Curved/graded surface rasterization may fail this gate.
    """
    report={}
    for name,profile in zip(("narrow","wide"),preprocessor.profiles):
        transform=OPTICAL_FROM_CARLA @ images[name+"_rgb_camera_from_world"]
        points=getattr(reference,'depth_fixture_points',np.concatenate([reference.left_xyz,reference.right_xyz,reference.center_xyz]))
        pixels,optical=project_world_points(points,transform,profile.K)
        # Preselect by geometry only. Never discard a depth residual before
        # scoring the fixture: doing so would hide alignment/unit failures.
        selected=(optical[:,2]>5)&(optical[:,2]<10)&(np.abs(optical[:,0])>.6)
        measured=[]; coordinates=[]; source=[]
        for index in np.flatnonzero(selected):
            x,y=np.rint(pixels[index]).astype(int)
            if 3<=x<profile.width-3 and 3<=y<profile.height-3:
                depth=float(images[name+"_depth"][y,x])
                measured.append(depth);coordinates.append(optical[index]);source.append(dict(world=points[index].tolist(),pixel=[int(x),int(y)]))
        if len(measured)>=3:
            result=validate_depth_geometry(np.array(measured),np.array(coordinates),tolerance_m=.15)
        else:
            result=dict(qualified=False,reason="insufficient_off_axis_known_surface_samples",depth_mode=None)
        result.update(samples=source,measured_m=measured,camera_points=_finite_json(coordinates),fixture="collision-surveyed road interiors" if hasattr(reference,'depth_fixture_points') else "OpenDRIVE lane boundaries and center at matching road arc",camera_from_world=transform.tolist())
        report[name]=result
    return report


def _longest_supported(mask, near_index):
    indices=np.flatnonzero(mask)
    runs=np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1)
    eligible=[r for r in runs if len(r)>=4 and r[-1]>=near_index]
    if not eligible:
        raise ValueError("no_contiguous_paired_boundary_observations")
    # Prefer the earliest visible corridor ahead of the ego, not isolated far road.
    return min(eligible,key=lambda r:abs(int(r[0])-near_index))


def observed_reference(reference, support, position, timestamp, maximum_marking_gap_m=MAX_OBSERVED_KNOT_GAP_M):
    from .observed_corridor import reconstruct
    return reconstruct(reference,support,position,timestamp,maximum_marking_gap_m)


def longitudinal_setpoint(previous, acceleration, target_speed, calibration, *, recording, dt=.05):
    """Keep the executed cruise reference inside the MPC's measured speed domain."""
    lower=calibration.speed_min_mps if recording else 0.
    upper=min(target_speed,calibration.speed_max_mps) if recording else target_speed
    if upper<lower:
        raise ValueError('planned_cruise_requires_uncalibrated_speed')
    return float(np.clip(previous+acceleration*dt,lower,upper))


def capture_development(root,port=2500,worker=1,ticks=120,benchmark=False,task=None):
    # Ownership includes sensor/actor cleanup and shard finalization, so a new
    # process cannot reload this world while the preceding run is destroying it.
    with world_owner(port,root):
        return _capture_owned(root,port,worker,ticks,benchmark,task)


def recording_budget(task, *, warmup_progress_m, recording_start_m, route_end_m,
                     ticks, maximum_speed_mps):
    """Check declared route reserves against the actual rolling initialization."""
    if task is None or task.get("warmup_distance_budget_m") is None:
        return None
    warmup=float(task["warmup_distance_budget_m"])
    preview=float(task["planned_preview_m"])
    values=(warmup,preview,warmup_progress_m,recording_start_m,route_end_m,maximum_speed_mps)
    if not all(math.isfinite(value) for value in values) or min(warmup,preview,maximum_speed_mps)<=0 or ticks<4:
        raise ValueError("capture requires a finite positive recording distance budget")
    if warmup_progress_m>warmup+1e-6:
        raise ValueError("physical rolling initialization exceeded planned distance budget")
    needed=ticks*Config().dt*maximum_speed_mps+preview
    remaining=route_end_m-recording_start_m
    if needed>remaining+1e-6:
        raise ValueError("remaining route cannot contain recording and forward preview")
    return dict(warmup_progress_m=float(warmup_progress_m),warmup_distance_budget_m=warmup,
                recording_start_m=float(recording_start_m),remaining_route_m=float(remaining),
                required_remaining_route_m=float(needed),planned_preview_m=preview,
                speed_bound_mps=float(maximum_speed_mps),passed=True)


def _capture_owned(root,port,worker,ticks,benchmark,task):
    import carla
    from PIL import Image
    if ticks<4:
        raise ValueError("capture requires at least four ticks")
    terminal_stop=validate_terminal_stop(task)
    recovery_offset=float((task or {}).get('initial_recovery_offset_m',0.))
    if recovery_offset and (not (task or {}).get('recovery') or abs(recovery_offset)>.3):
        raise ValueError('held initial offset is only allowed for a declared recovery episode')
    root=Path(root); config=Config(); store=Store(root,config)
    prep=StockPreprocessor(); archive=reference_module("frame_archive_v2")
    calibration,binding=_calibration(root,(task or {}).get("nominal_speed_mps",5.))
    episode=(task or {}).get("episode_id",f"{'benchmark' if benchmark else 'development'}-w{worker}-{time.time_ns()}")
    bound=2*archive.archive_worst_case_bytes(dtype=np.uint8,frame_shape=(6,128,256),expected_count=ticks)+256*1024**2
    token,out=store.reserve(episode,worker,bound)
    manifest=dict(schema_version=1,status="development",episode_id=episode,geometry_group_id=(task or {}).get("group_id","development-town04-r38"),
                  split=(task or {}).get("split","development"),family=(task or {}).get("family","straight_recovery"),
                  worker=worker,port=port,seed=(task or {}).get("seed",config.seed+worker),
                  appearance=(task or {}).get("appearance",0),direction=(task or {}).get("direction","straight"),procedural=(task or {}).get("procedural",False),
                  intended_progress_m=(task or {}).get("intended_progress_m",.8*ticks*.05*(task or {}).get("nominal_speed_mps",5.)),maneuver_phase_ranges=(task or {}).get("maneuver_phase_ranges"),
                  intent_required_indices=(task or {}).get("intent_required_indices"),
                  intent_required_route_range_m=(task or {}).get("intent_required_route_range_m"),
                  desire_route_range_m=(task or {}).get("desire_route_range_m"),
                  annotation_status=(task or {}).get("annotation_status"),
                  recovery=bool((task or {}).get("recovery",False)),nominal_speed_mps=(task or {}).get("nominal_speed_mps",5.),
                  recovery_onset_s=5. if recovery_offset else 0.,
                  config_sha256=config.digest(),calibration=binding,preprocessing=prep.contract(),
                  camera_optics=config.camera_optics_status,frames_file="frames.jsonl",labels_file="labels.npz",evidence_file="evidence.jsonl.gz",candidates_file="candidates.json.gz",command_provenance_file="command-provenance.jsonl.gz",
                  qa=dict(passed=False),audit_frames=[],accepted_label_count=0)
    write_json(out/"manifest.json",manifest)
    if task is not None:
        write_json(out/"requested-task.json",_finite_json(task))
    (out/"executed-sources").mkdir()
    manifest["source_hashes"]={}
    for source in sorted((PROJECT/"lane_dataset").glob("*.py")):
        shutil.copy2(source,out/"executed-sources"/source.name)
        manifest["source_hashes"][str(source)]=sha256_file(out/"executed-sources"/source.name)
    rows=[];bootstrap_rows=[];bootstrap=None;candidates=[];evidence_counts=[];timing=[];actors=[];writers={};rig=None;traffic=None
    boundary_frames=[];rejected_control_frames=0;terminal_stopping=False
    collisions=[];all_command_reasons=[];depth_report={};started=time.perf_counter()
    try:
        client=connect(port)
        if task and task.get("procedural"):
            world,settings=configure_world(client,"Town04",render=True)
            opendrive_path=Path(task["opendrive_path"])
            if sha256_file(opendrive_path)!=task["route"]["source_xodr_sha256"]:
                raise RuntimeError("procedural OpenDRIVE source hash mismatch")
            world=client.generate_opendrive_world(opendrive_path.read_text(),carla.OpendriveGenerationParameters(vertex_distance=.5,max_road_length=50.,wall_height=0.,additional_width=.6,smooth_junctions=True,enable_mesh_visibility=True))
            settings=apply_checked_settings(world,settings)
        else:
            world,settings=configure_world(client,(task or {}).get("town","Town04"),render=True)
        weather=carla.WeatherParameters.ClearNoon
        if (task or {}).get("appearance",0)%2:
            weather=carla.WeatherParameters.WetCloudySunset
            weather.precipitation=25.;weather.wetness=70.
        world.set_weather(weather)
        waypoint=world.get_map().get_waypoint_xodr(**task["start"]) if task and "start" in task else world.get_map().get_waypoint_xodr(38,4,200.)
        if waypoint is None: raise RuntimeError("episode start waypoint unavailable")
        if task and "route" in task:
            route=task["route"]
        else:
            # The development route includes the maximum physical warmup
            # budget, dense recording, and a preview margin. Planned task
            # routes remain fixed so their split envelopes cannot expand.
            route_distance=max(190.,manifest["nominal_speed_mps"]*(ticks*.05+20.)+25.)
            points=sample_route(waypoint,route_distance,.5,(task or {}).get("branch",0))
            section,route=route_document(points,(task or {}).get("town","Town04"))
        if task and task.get("survey_road_surface",False):
            route,surface_report=survey_road_surface(world,route)
            write_json(out/"road-surface-survey.json",surface_report)
        reference=LaneReference(route["left_xyz"],route["right_xyz"],route["arc_m"])
        bind_marking_widths(reference,route)
        if 'surface_fixture_xyz' in route:
            reference.depth_fixture_points=np.asarray(route['surface_fixture_xyz'],float)
        write_json(out/"geometry.json",route)
        ego=spawn_vehicle(world,waypoint,(task or {}).get("offset_m",0),(task or {}).get("heading_error_deg",0));actors.append(ego)
        box_rotation=ego.bounding_box.rotation
        box_rpy=[box_rotation.roll,box_rotation.pitch,box_rotation.yaw]
        if not np.allclose(box_rpy,0.,atol=1e-7,rtol=0.):
            raise RuntimeError("vehicle actor-local bounding-box rotation must be zero for this rig")
        collision=world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
        collision.listen(lambda event:collisions.append(dict(frame=event.frame,other_actor=event.other_actor.id)));actors.append(collision)
        # Settle using physics. No per-frame pose or velocity replacement.
        batch_control(client,ego,brake=1.)
        for _ in range(30): world.tick()
        traffic_count=int((task or {}).get("traffic_count",0))
        if traffic_count:
            traffic=SurroundingTraffic.spawn(client,world,ego,reference,worker=worker,rpc_port=port,
                seed=manifest["seed"],config=TrafficConfig(count=traffic_count,speed_mps=manifest["nominal_speed_mps"]))
        manifest["traffic_count"]=traffic_count
        semantic_qa=bool((task or {}).get("semantic_qa",False))
        rig=CameraRig(world,ego,semantic_qa=semantic_qa)
        manifest.update(physics_readback=settings,camera_sensors=rig.specs,server_version=client.get_server_version(),client_version=client.get_client_version(),
                        vehicle_type=ego.type_id,vehicle_dimensions=dict(length=2*ego.bounding_box.extent.x,width=2*ego.bounding_box.extent.y,height=2*ego.bounding_box.extent.z,bbox_center=[ego.bounding_box.location.x,ego.bounding_box.location.y,ego.bounding_box.location.z],bbox_rotation_degrees=box_rpy))
        writers={name:archive.FrameArchiveWriter(out/f"packed_{name}.npz",dtype=np.uint8,frame_shape=(6,128,256),expected_count=ticks) for name in ("narrow","wide")}
        history=EvidenceHistory();n=len(reference.s)
        point_ids=tuple([f"L{i}" for i in range(n)]+[f"R{i}" for i in range(n)])
        evidence_points=np.concatenate([reference.left_xyz,reference.right_xyz]);arc=np.tile(reference.s,2)
        evidence_normals=surface_normals(reference)
        mpc=BicycleMPC(calibration,MPCConfig(require_calibration=False,vehicle_width_m=2*ego.bounding_box.extent.y,vehicle_length_m=2*ego.bounding_box.extent.x))
        integral=0.;preview_ready=False;longitudinal_reference=0.;previous_throttle=0.;previous_brake=1.
        bootstrap=RollingWarmup(calibration.speed_min_mps,calibration.speed_max_mps)
        initializer=PhysicalInitializer(calibration);map_initialization_frames=0
        if not calibration.qualified:raise RuntimeError("measured steering calibration is unqualified")
        recording=False;index=0
        audit_ticks=set(np.linspace(0,ticks-1,5).astype(int).tolist())
        (out/"audit").mkdir()
        with (out/"frames.jsonl").open("w",encoding="utf-8") as frame_file, gzip.open(out/"evidence.jsonl.gz","wt",encoding="utf-8",compresslevel=1) as evidence_file, gzip.open(out/"command-provenance.jsonl.gz","wt",encoding="utf-8",compresslevel=1) as command_file:
            for simulation_index in range(ticks+bootstrap.maximum_frames):
                clock=time.perf_counter();frame=world.tick();snapshot=world.get_snapshot()
                if snapshot.frame!=frame: raise RuntimeError("world snapshot frame mismatch")
                if traffic is not None:traffic.check(snapshot)
                images=rig.read(snapshot);sensor_done=time.perf_counter()
                converted=[image_to_nv12(images[cam+"_rgb"],"RGB") for cam in ("narrow","wide")]
                packed=prep.pack_nv12(converted[0][0],converted[1][0],(converted[0][1],converted[1][1]));packed_done=time.perf_counter()
                planes_by_camera={cam:split_nv12(*converted[i]) for i,cam in enumerate(("narrow","wide"))}
                luma_by_camera={cam:planes[0] for cam,planes in planes_by_camera.items()}
                row=state(ego,snapshot);current_s,_=reference.project(row["position_world"][:2]);near=int(np.searchsorted(reference.s,current_s))
                if simulation_index==0:
                    depth_report=_depth_fixture(images,prep,reference)
                    write_json(out/"depth-fixture.json",_finite_json(depth_report))
                    if not all(value["qualified"] for value in depth_report.values()):
                        raise RuntimeError("known geometry depth fixture failed before rolling initialization")
                support=None;command=None;observable=None;reason=[]
                evidence_documents=[]
                if all(value["qualified"] for value in depth_report.values()):
                    for cam,profile,warp in zip(("narrow","wide"),prep.profiles,prep.warps):
                        observation=CameraObservation(frame,row["timestamp"],cam,images[cam+"_rgb"],images[cam+"_depth"],
                            OPTICAL_FROM_CARLA@images[cam+"_rgb_camera_from_world"],profile.K,warp,depth_mode=depth_report[cam]["depth_mode"],depth_validated=True,luma=luma_by_camera[cam],
                            chroma_u=planes_by_camera[cam][1],chroma_v=planes_by_camera[cam][2],
                            surface_normals_world=evidence_normals,
                            marking_mask=(images[cam+"_semantic_segmentation"]==int(carla.CityObjectLabel.RoadLines)) if semantic_qa else None,
                            road_mask=np.isin(images[cam+"_semantic_segmentation"],[int(carla.CityObjectLabel.Roads),int(carla.CityObjectLabel.RoadLines)]) if semantic_qa else None,
                            road_exterior_mask=np.isin(images[cam+"_semantic_segmentation"],[int(carla.CityObjectLabel.Sidewalks),int(carla.CityObjectLabel.Terrain),int(carla.CityObjectLabel.Ground)]) if semantic_qa else None)
                        evidence=probe_lane_evidence(observation,evidence_points,point_ids,arc)
                        history.admit(evidence)
                        # Compact full point flags with raw pixel/depth checks at every tick.
                        evidence_documents.append(evidence.compact())
                    required=(arc>=current_s)&(arc<=current_s+max(3.,row["speed"]*.5))
                    support=history.support(frame,row["timestamp"],point_ids,required,np.maximum(0,arc-current_s)/max(row["speed"],1.),
                        [point_ids[i] for i in np.flatnonzero(required)],distance_m=arc-current_s)
                    paired=support.supported[:n]&support.supported[n:]
                    terminal_stopping=terminal_stopping or bool(recording and terminal_stop is not None and current_s>=terminal_stop)
                    try:
                        if terminal_stopping:
                            raise ValueError('planned_unlabelled_terminal_suffix_stop')
                        observable=observed_reference(reference,support,row["position_world"],row["timestamp"])
                        yaw=row["rotation_world"][2];acceleration=row["acceleration_world"]
                        measured_lateral=-math.sin(yaw)*acceleration[0]+math.cos(yaw)*acceleration[1]
                        vehicle_state=VehicleState(row["position_world"][0],row["position_world"][1],yaw,row["speed"],row["angular_velocity_world"][2],timestamp_s=row["timestamp"],lateral_acceleration_mps2=measured_lateral)
                        desired_speed=min((task or {}).get("nominal_speed_mps",5.),max(0.,simulation_index*.05*.8))
                        required_knots=np.zeros(2*n,bool)
                        required_knots[observable.source_point_indices]=True
                        support=history.support(frame,row["timestamp"],point_ids,required_knots,np.maximum(0,arc-current_s)/max(row["speed"],1.),
                            [point_ids[i] for i in np.flatnonzero(required_knots)],distance_m=arc-current_s)
                        held_offset=recovery_offset if not recording or index<100 else 0.
                        command=mpc.control(vehicle_state,observable,desired_speed,desired_offset_m=held_offset)
                        reason.extend(command.reasons)
                        if command.accepted:
                            longitudinal_reference=longitudinal_setpoint(longitudinal_reference,command.acceleration_mps2,
                                command.target_speed_mps,calibration,recording=recording)
                            error=longitudinal_reference-row["speed"];integral=float(np.clip(integral+error*.05,-2,2))
                            demand=.28*error+.08*integral+.1
                            throttle=float(np.clip(max(0,demand),max(0,previous_throttle-.02),previous_throttle+.02))
                            brake=float(np.clip(max(0,-demand),max(0,previous_brake-.05),previous_brake+.05))
                            if brake>0: throttle=0.
                            batch_control(client,ego,throttle=throttle,brake=brake,steer=command.carla_steer)
                            previous_throttle,previous_brake=throttle,brake
                            preview_ready=True
                        else:
                            previous_throttle,previous_brake=0.,min(.5,float(row['brake'])+.05)
                            longitudinal_reference=row["speed"]
                            batch_control(client,ego,brake=previous_brake,steer=row['steer'])
                    except ValueError as error:
                        # A fallback brake must never retain an accepted expert
                        # command or the longitudinal state of an unsent action.
                        command=None;observable=None
                        stop_control=suffix_stop_control(row)
                        previous_throttle,previous_brake=stop_control['throttle'],stop_control['brake']
                        longitudinal_reference=row["speed"];integral=0.
                        reason.append(str(error));batch_control(client,ego,**stop_control)
                else:
                    reason.append("depth_fixture_not_qualified");batch_control(client,ego,brake=1.)
                if (not recording and (task or {}).get('physical_map_initialization',False)
                        and (not (command and command.accepted) or row['speed']<.8*manifest['nominal_speed_mps'])):
                    # Unretained spawn initialization is explicitly separate from
                    # the pixel-only expert. It never supplies an accepted command
                    # or a label, and cannot make the rolling warmup ready.
                    applied=initializer.control(row,reference,min(manifest['nominal_speed_mps'],simulation_index*.05*.5),recording=recording)
                    batch_control(client,ego,**applied)
                    previous_throttle,previous_brake=applied['throttle'],applied['brake']
                    mpc.reset()
                    command=None;reason.append('physical_rolling_speed_initialization')
                    longitudinal_reference=row['speed'];integral=0.
                    map_initialization_frames+=1
                    manifest['map_initialization_frames']=map_initialization_frames
                    manifest['initialization_policy']='map-based physical spawn initialization allowed only before recording; readiness requires steady accepted pixel-only controls'
                controlled=time.perf_counter()
                row.update(command_curvature=0. if command is None else command.curvature_1pm,
                           command_steer=0. if command is None else command.carla_steer,
                           command_acceleration=0. if command is None else command.acceleration_mps2,
                           longitudinal_reference=longitudinal_reference,
                           command_accepted=bool(command and command.accepted),command_reasons=reason,route_s=current_s)
                rejected_control_frames=0 if command and command.accepted else rejected_control_frames+1
                rejection_limit=(task or {}).get('maximum_control_rejection_frames')
                if recording and not terminal_stopping and rejection_limit is not None and rejected_control_frames>=int(rejection_limit):
                    raise RuntimeError(f'sustained control rejection at route {current_s:.3f} m: {reason}')
                if (root/'reports'/('stop-'+episode+'.txt')).exists():
                    raise RuntimeError('capture stopped by explicit stop file')
                if not recording:
                    bootstrap_rows.append(row)
                    yaw=row["rotation_world"][2];acc=row["acceleration_world"]
                    lateral=-math.sin(yaw)*acc[0]+math.cos(yaw)*acc[1]
                    if collisions:raise RuntimeError("collision during rolling initialization")
                    recording=bootstrap.observe(row["timestamp"],row["speed"],lateral,
                        command_accepted=bool(command and command.accepted),visible_reference=observable is not None)
                    if recording:
                        manifest["bootstrap"]=bootstrap.report()
                        manifest["bootstrap"]["route_progress_m"]=float(current_s-bootstrap_rows[0]["route_s"])
                        manifest["bootstrap"]["last_frame_id"]=frame
                        manifest["bootstrap"]["last_timestamp"]=row["timestamp"]
                    if simulation_index%20==0 or recording:
                        store.check(token)
                        print(json.dumps(dict(episode=episode,phase="rolling_initialization",frame=simulation_index,
                            speed=row["speed"],ready=recording,steady_intervals=bootstrap.consecutive_intervals,reasons=reason)),flush=True)
                    continue
                if index==0:
                    if terminal_stop is not None:
                        manifest['terminal_suffix_policy']=dict(stop_route_m=terminal_stop,
                            expert_labels=False,method='physical_braking_after_complete_maneuver; causal supervision masked before stop controls')
                        manifest['intended_progress_m']=float(task['maneuver_phase_ranges']['recovery'][1]-current_s)
                        if manifest['intended_progress_m']<=0:
                            raise ValueError('recording starts after the maneuver recovery endpoint')
                    manifest["recording_distance_budget"]=recording_budget(task,
                        warmup_progress_m=manifest["bootstrap"]["route_progress_m"],
                        recording_start_m=current_s,route_end_m=float(reference.s[-1]),
                        ticks=ticks,maximum_speed_mps=calibration.speed_max_mps)
                    seam=check_recording_seam(bootstrap_rows[-1],row)
                    manifest["bootstrap"]["recording_seam"]=seam
                    if not seam["passed"]:
                        raise RuntimeError("physical recording seam failed: "+",".join(seam["errors"]))
                budget=manifest.get("recording_distance_budget")
                if budget is not None and current_s+budget["planned_preview_m"]>reference.s[-1]:
                    raise RuntimeError("recording reached its reserved forward-preview boundary")
                if semantic_qa:
                    boundary_frames.append(dict(frame_id=frame,index=index,cameras={
                        cam:measure_camera(reference=reference,current_s=current_s,
                            tags=images[cam+"_semantic_segmentation"],depth=images[cam+"_depth"],
                            camera_from_world=OPTICAL_FROM_CARLA@images[cam+"_rgb_camera_from_world"],
                            intrinsics=profile.K,depth_mode=depth_report[cam]["depth_mode"],
                            road_line_tag=int(carla.CityObjectLabel.RoadLines),luma=luma_by_camera[cam],
                            road_tag=int(carla.CityObjectLabel.Roads),
                            exterior_tags=[int(carla.CityObjectLabel.Sidewalks),int(carla.CityObjectLabel.Terrain),int(carla.CityObjectLabel.Ground)])
                        for cam,profile in zip(("narrow","wide"),prep.profiles)}))
                for document in evidence_documents:
                    evidence_file.write(json.dumps(_finite_json(document),separators=(",",":"))+"\n")
                command_file.write(json.dumps(dict(index=index,frame_id=frame,timestamp=row["timestamp"],
                    command_accepted=bool(command and command.accepted),
                    optimizer_success=None if command is None else command.optimizer_success,
                    optimizer_message=None if command is None else command.optimizer_message,
                    min_constraint_margin=None if command is None else command.min_constraint_margin,
                    constraint_margins=None if command is None else command.constraint_margins,
                    target_speed_mps=None if command is None else command.target_speed_mps,
                    source_indices=[] if observable is None else observable.source_indices.tolist(),
                    source_point_indices=[] if observable is None else observable.source_point_indices.tolist(),
                    evidence_frames=[] if command is None else [asdict(ref) for ref in command.evidence_frames]),separators=(",",":"))+"\n")
                for cam,value in zip(("narrow","wide"),packed): writers[cam].write(index,value)
                desire=route_desire(task,index,current_s)
                metadata=dict(index=index,frame_id=frame,timestamp=row["timestamp"],desire=desire,traffic_convention=[1.,0.],action_t=[calibration.delay_s+.05,.35])
                frame_file.write(json.dumps(metadata,separators=(",",":"))+"\n")
                rows.append(row);all_command_reasons.extend(reason)
                evidence_counts.append(dict(index=index,paired_supported=0 if support is None else int(np.sum(support.supported[:n]&support.supported[n:])),
                                            category="diagnostic_ambiguous" if support is None else support.category))
                if index>=100 and index%10==0 and index<=ticks-201:
                    candidates.append(dict(index=index,support=None if support is None else support.compact(),command_valid=bool(command and command.accepted and calibration.qualified and calibration.speed_min_mps<=row["speed"]<=calibration.speed_max_mps and calibration.steering_fraction_min<=command.carla_steer<=calibration.steering_fraction_max),
                                           command_evidence=[] if command is None else [asdict(ref) for ref in command.evidence_frames],
                                           interpolation_knot_indices=[] if observable is None else observable.source_indices.tolist(),
                                           observable_reference=None if observable is None else observable_provenance(observable)))
                if index in audit_ticks:
                    files={}
                    for cam in ("narrow","wide"):
                        relative=f"audit/{cam}-{index:05d}.png";Image.fromarray(images[cam+"_rgb"]).save(out/relative);files[cam]=relative
                    manifest["audit_frames"].append(dict(index=index,frame_id=frame,timestamp=row["timestamp"],files=files))
                if index%20==0: store.check(token)
                timing.append(dict(sensor_s=sensor_done-clock,packing_s=packed_done-sensor_done,visibility_controller_s=controlled-packed_done,total_s=time.perf_counter()-clock))
                if index%20==0 or (reason and not terminal_stopping): print(json.dumps(dict(episode=episode,tick=index,ticks=ticks,speed=row["speed"],route_s=current_s,supported=evidence_counts[-1]["paired_supported"],reasons=reason,
                    observed_route_arc=None if observable is None else reference.s[observable.source_indices[[0,-1]]].tolist(),
                    optimizer_message=None if command is None else command.optimizer_message,
                    min_constraint_margin=None if command is None else command.min_constraint_margin,
                    constraint_margins=None if command is None or not reason else command.constraint_margins)),flush=True)
                index+=1
                if index>=ticks:break
        for writer in writers.values(): writer.finalize(complete=True)
        arrays={key:np.asarray([row[key] for row in rows]) for key in ("frame_id","timestamp","position_world","rotation_world","velocity_world","acceleration_world","angular_velocity_world","speed","steer","throttle","brake","gear","hand_brake","reverse","manual_gear_shift","command_curvature","command_steer","command_acceleration","longitudinal_reference","command_accepted","route_s")}
        np.savez_compressed(out/"telemetry.npz",**arrays)
        if recovery_offset:
            edges=[100,140,180,220,260,300]
            if len(rows)<=edges[-1]:raise ValueError('recovery recording lacks complete measured event phases')
            manifest['maneuver_phase_ranges']={phase:[float(arrays['route_s'][a]),float(arrays['route_s'][b])]
                for phase,a,b in zip(('approach','entry','apex','exit','recovery'),edges,edges[1:])}
            manifest['recovery_event']=dict(onset_index=100,onset_s=float(arrays['timestamp'][100]-arrays['timestamp'][0]),
                held_offset_m=recovery_offset,physical_offset_at_release_m=float(reference.project(arrays['position_world'][100,:2])[1]),
                method='observed-boundary MPC held initial offset through recurrent prefix; center target released at index 100; no pose replacement')
            if abs(manifest['recovery_event']['physical_offset_at_release_m'])<.15:
                raise ValueError('recovery episode did not establish the intended physical initial offset')
            manifest['intended_progress_m']=float(arrays['route_s'][300]-arrays['route_s'][0])
        with gzip.open(out/"candidates.json.gz","wt",encoding="utf-8",compresslevel=1) as stream:
            json.dump(_finite_json(candidates),stream,separators=(",",":"),allow_nan=False)
        write_json(out/"evidence-summary.json",evidence_counts)
        yaw=arrays["rotation_world"][:,2];acc=arrays["acceleration_world"]
        lateral=-np.sin(yaw)*acc[:,0]+np.cos(yaw)*acc[:,1]
        qa=qualify_episode(timestamps_s=arrays["timestamp"],positions_xyz=arrays["position_world"],yaw_rad=yaw,speeds_mps=arrays["speed"],lateral_acceleration_mps2=lateral,
            steering_fractions=arrays["steer"],reference=reference,vehicle_length_m=2*ego.bounding_box.extent.x,vehicle_width_m=2*ego.bounding_box.extent.y,
            collisions=len(collisions),rendered_boundary_agreement=None,completed_maneuver=arrays["route_s"][-1]-arrays["route_s"][0]>=manifest["intended_progress_m"],intended_progress_m=manifest["intended_progress_m"],
            recovery=bool((task or {}).get("recovery",False)),recovery_onset_s=manifest['recovery_onset_s'],velocity_world=arrays["velocity_world"],rotation_world=arrays["rotation_world"],vehicle_height_m=2*ego.bounding_box.extent.z,vehicle_bbox_center_xyz=manifest["vehicle_dimensions"]["bbox_center"])
        manifest["qa"]=dict(passed=False,physical=qa.to_dict(),pending=["frozen_configuration_and_splits","20_observable_labels","per_geometry_visual_review","complete_maneuver"])
        manifest["archives"]={name:dict(path=f"packed_{name}.npz",sha256=sha256_file(out/f"packed_{name}.npz")) for name in writers}
        manifest["depth_qualified"]=all(v["qualified"] for v in depth_report.values())
        manifest["measured_throughput"]=dict(ticks=ticks,simulated_seconds=ticks*.05,wall_seconds=time.perf_counter()-started,
            steady_tick_seconds=float(np.mean([t["total_s"] for t in timing])),component_mean_seconds={k:float(np.mean([t[k] for t in timing])) for k in timing[0]},
            initialization_frames=len(bootstrap_rows),total_simulator_frames=len(bootstrap_rows)+ticks,
            scope="development full RGB/depth/stock packing/pixel evidence/MPC/lossless archives, including physical rolling initialization cost",complete_maneuver_qualified=False)
        manifest["command_rejections"]=dict(__import__("collections").Counter(all_command_reasons))
        manifest["collision_events"]=collisions
    except BaseException as error:
        manifest["error"]=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        manifest['collision_events']=collisions
        if traffic is not None:
            traffic_errors=traffic.close()
            manifest["traffic"]=traffic.summary()
            manifest["traffic"]["cleanup_errors"]=traffic_errors
            if traffic_errors or not manifest["traffic"]["passed"]:
                manifest.setdefault("error",dict(type="TrafficRejected",message="surrounding traffic QA or cleanup failed"))
            write_json(out/"traffic-report.json",_finite_json(manifest["traffic"]))
        if rig is not None: rig.close()
        for writer in writers.values(): writer.close()
        for actor in reversed(actors):
            try: actor.destroy()
            except RuntimeError: pass
        if bootstrap is not None:
            report=bootstrap.report()
            report["control_state_semantics"]="throttle/brake/steer/gear were read with each recorded physical snapshot; command_* apply after that snapshot"
            write_json(out/"bootstrap-report.json",_finite_json(report))
            manifest["bootstrap"]={**manifest.get("bootstrap",{}),**report}
        if bootstrap_rows:
            bootstrap_arrays={key:np.asarray([row[key] for row in bootstrap_rows]) for key in bootstrap_rows[0] if key!="command_reasons"}
            np.savez_compressed(out/"bootstrap-telemetry.npz",**bootstrap_arrays)
        if rows and not (out/'telemetry.npz').exists():
            partial={key:np.asarray([row[key] for row in rows]) for key in rows[0] if key!='command_reasons'}
            np.savez_compressed(out/'partial-telemetry.npz',**partial)
            manifest['partial_recording_frames']=len(rows)
        if boundary_frames:
            with gzip.open(out/"rendered-boundary-samples.json.gz","wt",encoding="utf-8",compresslevel=1) as stream:
                json.dump(boundary_frames,stream,separators=(",",":"),allow_nan=False)
            boundary_report=summarize_frames(boundary_frames)
            boundary_report["samples_sha256"]=sha256_file(out/"rendered-boundary-samples.json.gz")
            write_json(out/"rendered-boundary-report.json",boundary_report)
        manifest["files"]={str(p.relative_to(out)).replace("\\","/"):sha256_file(p) for p in out.rglob("*") if p.is_file() and p.name!="manifest.json"}
        write_json(out/"manifest.json",_finite_json(manifest))
        destination=store.finish(token,False,"development_capture_not_release_qualified")
    result=dict(path=str(destination),status="quarantined_development",manifest=manifest)
    write_json(root/"reports"/(episode+".json"),_finite_json(result))
    return result


def generate(root,port,worker,worker_count=2):
    """Resume only after the independent development qualification is sealed."""
    if worker_count not in (1,2) or not 0<=worker<worker_count:raise ValueError("worker must belong to active one/two-worker partition")
    root=Path(root);store=Store(root);store.recover()
    gate=root/"qualification.json"
    if not gate.is_file():
        raise RuntimeError("bulk generation blocked: no sealed development qualification; inspect reports/validation.json and run generate --development")
    from .admission import verify_qualification, qualify_candidate
    tasks_path=root/"reports/scenario-tasks.json"
    tasks=json.loads(tasks_path.read_text())
    worker_tasks=[task for index,task in enumerate(tasks) if index%worker_count==worker]
    reports=[]
    for task in worker_tasks:
        if (root/"accepted"/task["episode_id"]).exists(): continue
        probe=dict(config_sha256=Config().digest(),geometry_group_id=task["group_id"],split=task["split"],family=task["family"])
        qualified=verify_qualification(root,probe)
        if not qualified["passed"]:
            raise RuntimeError(f"bulk generation qualification failed: {qualified['errors']}")
        attempts=[row for row in store.rows() if row["episode"]==task["episode_id"]]
        if len(attempts)>=Config().max_attempts:
            reports.append(dict(episode_id=task["episode_id"],status="retry_limit_reached"));continue
        captured=capture_development(root,port,worker,max(500,task["minimum_ticks"]),task=task)
        shard=Path(captured["path"])
        if captured["manifest"].get("error"):
            reports.append(dict(episode_id=task["episode_id"],status="capture_failed",error=captured["manifest"]["error"]));continue
        admission=qualify_candidate(shard,dataset_root=root)
        if admission["passed"]:
            destination=store.publish_qualified(shard)
            reports.append(dict(episode_id=task["episode_id"],status="accepted",path=str(destination)))
        else:
            reports.append(dict(episode_id=task["episode_id"],status="diagnostic_not_admitted",errors=admission["errors"]))
    result=dict(worker=worker,episodes=reports,accepted_episodes=sum(row["status"]=="accepted" for row in reports))
    write_json(root/"reports"/f"generation-worker-{worker}-{time.time_ns()}.json",result)
    return result
