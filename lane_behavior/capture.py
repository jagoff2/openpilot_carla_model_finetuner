"""Physics-driven native unmarked-road demonstrations, prior to admission.

The privileged teacher controls the simulator only. Camera-derived supervision
and independent physical admission are separate and must be replayed before a
recording enters training. Nothing in this module publishes accepted labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import traceback
import uuid

import numpy as np

from lane_dataset.capture import _calibration, _finite_json
from lane_dataset.config import Config, sha256_file, write_json
from lane_dataset.controller import VehicleState
from lane_dataset.depth_fixture import collision_depth_fixture
from lane_dataset.geometry import LaneReference
from lane_dataset.preprocessing import StockPreprocessor, reference_module, image_to_nv12, split_nv12
from lane_dataset.rendered_boundaries import measure_camera, summarize_frames
from lane_dataset.render_quality import apply_production_weather
from lane_dataset.simulator import (world_owner, connect, configure_world, spawn_vehicle,
    batch_control, state, apply_checked_settings)
from lane_dataset.traffic import convex_hull
from .planned import PlannedObstacle, planned_scene_geometry
from .target import EgoFootprint
from .qualification_probe import (ProofCameraRig, OPTICAL_FROM_CARLA, load_probe_task,
    actor_document, destroy_probe_actor)
from .grounding import spawn_grounded_obstacle
from .world_access import require_reserved_or_idle


def capture_bounds(task, ticks, start_s):
    if task.get("native_unmarked") is not True or task.get("procedural") is not False:
        raise ValueError("physical behavior capture requires a native road candidate")
    if type(ticks) is not int or not 400 <= ticks <= 2400:
        raise ValueError("capture budget must be 400..2400 uninterrupted 20 Hz frames")
    if not math.isfinite(start_s) or not 5 <= start_s <= task["route"]["length_m"]-40:
        raise ValueError("capture start must retain road for approach, avoidance and recovery")
    # Raw proof payloads every five ticks, packed RGB every tick. This allowance
    # includes conservative raw compression overhead without deleting old data.
    return 512*1024**2 + ticks*(2*6*128*256 + 40*1024**2/5)


def controlled_scene_record(world, snapshot, ego, obstacle):
    """Bind the complete owned scene; this is QA, never visual free space."""
    permitted={int(ego.id)} | ({int(obstacle.id)} if obstacle is not None else set())
    traffic=set();inventory=[]
    for actor in sorted(world.get_actors(),key=lambda a:a.id):
        is_traffic=actor.type_id.startswith(("vehicle.","walker."))
        if is_traffic:traffic.add(int(actor.id))
        inventory.append(dict(actor_id=int(actor.id),type_id=actor.type_id,
            state=actor_document(actor,snapshot) if is_traffic else None))
    if traffic!=permitted:
        raise RuntimeError("controlled static scene has missing or unexpected traffic actors")
    return dict(schema="controlled_native_static_scene_v1",frame_id=int(snapshot.frame),
        timestamp_s=float(snapshot.timestamp.elapsed_seconds),ego_actor_id=int(ego.id),
        static_obstacle_actor_ids=[] if obstacle is None else [int(obstacle.id)],actors=inventory,
        scope="simulator_scene_QA_not_visual_free_space")


def save_evidence_frame(output, index, snapshot, images, stamps, packed, prep, ego, obstacle, *, controlled_scene=None):
    """Save exact model packing and raw modalities with same-frame identities."""
    record = dict(index=index, frame_id=int(snapshot.frame), timestamp_s=float(snapshot.timestamp.elapsed_seconds),
        scene_actor_ids=sorted(int(item.id) for item in snapshot), ego=actor_document(ego, snapshot),
        obstacle=actor_document(obstacle, snapshot) if obstacle else None, cameras={})
    if controlled_scene is not None:record["controlled_scene"]=controlled_scene
    for i, (camera, profile, warp) in enumerate(zip(("narrow", "wide"), prep.profiles, prep.warps)):
        path = output/f"raw/frame-{index:05d}-{camera}.npz"
        np.savez_compressed(path, rgb=images[camera+"_rgb"], depth_m=images[camera+"_depth"],
            semantic_ids=images[camera+"_instance_segmentation_semantic"],
            instance_ids=images[camera+"_instance_segmentation"],
            instance_bgra=images[camera+"_instance_segmentation_bgra"],
            depth_bgra=images[camera+"_depth_bgra"], semantic_camera_ids=images[camera+"_semantic_segmentation"],
            rgb_bgra=images[camera+"_rgb_bgra"], packed=packed[i], intrinsics=profile.K,
            camera_from_world=OPTICAL_FROM_CARLA@images[camera+"_rgb_camera_from_world"], model_to_source=warp)
        record["cameras"][camera] = dict(payload=path.relative_to(output).as_posix(),
            payload_sha256=sha256_file(path), modality_stamps={"rgb":stamps[camera+"_rgb"],
                "packed":stamps[camera+"_rgb"], "depth":stamps[camera+"_depth"],
                "semantic":stamps[camera+"_instance_segmentation"], "instance":stamps[camera+"_instance_segmentation"]},
            separate_semantic_sensor_stamp=stamps[camera+"_semantic_segmentation"])
    return record


def run_capture(*, port, root, plan, case, ticks=1200, start_s=10., paused_supervisor_pid=None):
    import carla
    from .teacher import PhysicalTeacher, TeacherConfig
    from .teacher_quality import qualify_teacher_rollout
    root = Path(root).resolve()
    task = load_probe_task(plan, case)
    required = capture_bounds(task, ticks, start_s)
    if shutil.disk_usage(root).free < required+10*1024**3:
        raise RuntimeError("insufficient disk headroom for bounded raw behavior evidence")
    require_reserved_or_idle(root, port, paused_supervisor_pid=paused_supervisor_pid)
    output = root/"reports/unmarked/captures"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8])
    report = dict(schema_version=1, status="capturing", scope="physical_teacher_capture_pending_observed_admission",
        training_ready=False, accepted_labels=0, port=port, episode_id=task["episode_id"],
        split="development", case=case, plan=str(Path(plan).resolve()), plan_sha256=sha256_file(plan),
        requested_ticks=ticks, errors=[], cleanup_errors=[], frame_records=[])
    with world_owner(port, root):
        require_reserved_or_idle(root, port, paused_supervisor_pid=paused_supervisor_pid)
        client = connect(port)
        previous = client.get_world()
        if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in previous.get_actors()):
            raise RuntimeError("refusing to replace another owner's actors")
        output.mkdir(parents=True, exist_ok=False)
        (output/"raw").mkdir()
        (output/"executed-sources").mkdir()
        for package in ("lane_behavior", "lane_dataset"):
            target_directory = output/"executed-sources"/package
            target_directory.mkdir()
            for source in (Path(__file__).parents[1]/package).glob("*.py"):
                shutil.copy2(source, target_directory/source.name)
        report["source_hashes"] = {p.relative_to(output/"executed-sources").as_posix():sha256_file(p)
            for p in (output/"executed-sources").rglob("*.py")}
        write_json(output/"candidate-task.json", task)
        report["candidate_task_sha256"] = sha256_file(output/"candidate-task.json")
        report["retained_scene_bindings"] = {}
        scene_sources = task["route"]["unmarked_road"].get("source_bindings", {})
        if scene_sources:
            (output/"scene-sources").mkdir()
        for name, binding in scene_sources.items():
            if not isinstance(name,str) or not name.replace("_", "").isalnum():
                raise ValueError("invalid scene source name")
            source = Path(binding["path"])
            if sha256_file(source) != binding["sha256"]:
                raise ValueError("scene source changed: "+name)
            relative = "scene-sources/"+name+source.suffix
            shutil.copy2(source,output/relative)
            if sha256_file(output/relative) != binding["sha256"]:
                raise ValueError("retained scene source mismatch: "+name)
            report["retained_scene_bindings"][name] = dict(path=relative,sha256=binding["sha256"],original_path=binding["path"])
        old = previous.get_settings()
        original = {k:getattr(old,k) for k in ("synchronous_mode", "fixed_delta_seconds", "substepping",
            "max_substep_delta_time", "max_substeps", "no_rendering_mode")}
        actors, writers, rows, commands, evidence, edge_frames, collisions = [], {}, [], [], [], [], []
        rig, world, target = None, None, None
        try:
            config = Config()
            apply_checked_settings(previous, dict(synchronous_mode=True, fixed_delta_seconds=config.dt,
                substepping=True, max_substep_delta_time=config.max_substep_delta_time,
                max_substeps=config.max_substeps, no_rendering_mode=False))
            world, settings = configure_world(client, task["town"], render=True)
            report["world_settings"] = settings
            report["render_profile"] = apply_production_weather(world, carla, 0)
            route = task["route"]
            corridor = LaneReference(route["left_xyz"], route["right_xyz"], route["arc_m"])
            edges = route["unmarked_road"]
            road = LaneReference(edges["road_left_edge_xyz"], edges["road_right_edge_xyz"], route["arc_m"])
            xyz = corridor.sample(start_s)[0]
            waypoint = world.get_map().get_waypoint(carla.Location(*map(float,xyz)))
            if waypoint is None or waypoint.lane_id != task["start"]["lane_id"]:
                raise RuntimeError("capture start did not resolve to planned right-hand lane")
            loc, right = waypoint.transform.location, waypoint.transform.get_right_vector()
            offset = float((xyz-np.array([loc.x,loc.y,loc.z]))@np.array([right.x,right.y,right.z]))
            ego = spawn_vehicle(world, waypoint, offset=offset)
            actors.append(ego)
            batch_control(client, ego, brake=1.)
            for _ in range(30): world.tick()
            obstacle, spec, grounding = spawn_grounded_obstacle(world,task,corridor,actors,road_reference=road)
            write_json(output/"obstacle-grounding.json",grounding)
            initial_scene=controlled_scene_record(world,world.get_snapshot(),ego,obstacle)
            write_json(output/"controlled-scene.json",initial_scene)
            report["controlled_scene_sha256"]=sha256_file(output/"controlled-scene.json")
            box = ego.bounding_box
            if abs(box.rotation.roll)+abs(box.rotation.pitch)>1e-6:
                raise RuntimeError("unsupported nonplanar ego bounding box")
            footprint = EgoFootprint(2*box.extent.x, 2*box.extent.y,
                (box.location.x,box.location.y), math.radians(box.rotation.yaw))
            obstacles = () if obstacle is None else (PlannedObstacle(str(obstacle.id),
                convex_hull(np.array([[v.x,v.y] for v in obstacle.bounding_box.get_world_vertices(obstacle.get_transform())])),
                spec["kind"], motion="static"),)
            passing_space=None
            if (spec is not None and spec["kind"]=="parked_vehicle" and task["unmarked_behavior"].get("passing_space_policy")
                    =="observed_clear_surface_during_static_parked_avoidance_only"):
                from .passing import PlannedPassingSpace,reference_fingerprint
                passing_space=PlannedPassingSpace(reference_sha256=reference_fingerprint(road),
                    allowed_intervals_m=((float(road.s[0]),float(road.s[-1])),),opposing_width_m=1.,
                    obstacle_ids=(str(obstacle.id),),controlled_scene_sha256=report["controlled_scene_sha256"])
            target = planned_scene_geometry(road, obstacles, footprint,
                source_metadata=dict(plan_sha256=report["plan_sha256"], task_sha256=report["candidate_task_sha256"]),
                obstacle_tracking_reserve_m=.10,passing_space=passing_space)
            write_json(output/"teacher-target.json", target.document())
            if not target.feasible:
                raise RuntimeError("teacher path infeasible: "+", ".join(target.reasons))
            if case != "clear_road" and not target.excursions:
                raise RuntimeError("obstacle case did not require avoidance; adjust candidate placement")
            if "sequence_budget" in task:
                from .sequence_guard import validate_teacher_budget
                report["sequence_budget_check"] = validate_teacher_budget(task,target,start_s=start_s,ticks=ticks)
            calibration, binding = _calibration(root, task["nominal_speed_mps"])
            report["calibration"] = binding
            teacher = PhysicalTeacher(target, calibration, config=TeacherConfig(cruise_speed_mps=task["nominal_speed_mps"]))
            collision = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),carla.Transform(),attach_to=ego)
            collision.listen(lambda event: collisions.append(dict(frame=int(event.frame), other_actor=int(event.other_actor.id))))
            actors.append(collision)
            rig, prep = ProofCameraRig(world,ego), StockPreprocessor()
            report["camera_specs"], report["preprocessing"] = rig.specs, prep.contract()
            report["teacher_contract_sha256"] = target.contract_sha256
            report["ego_physics_policy"] = "CARLA physics and controls every tick; no pose/velocity replacement"
            report["obstacle_physics_policy"] = (
                "parked vehicle physically settled under brake and handbrake wholly on road before sensors"
                if spec is not None and spec["kind"]=="parked_vehicle" else
                "independently grounded static pedestrian fixture before sensors" if spec is not None else "no obstacle")
            for _ in range(24):
                world.tick(); rig.read(world.get_snapshot())
            archive = reference_module("frame_archive_v2")
            report["archive_implementation"] = dict(path=str(Path(archive.__file__).resolve()),sha256=sha256_file(archive.__file__))
            fixture, attempts = None, []
            for attempt in range(10):
                world.tick()
                images,_ = rig.read(world.get_snapshot())
                fixture = collision_depth_fixture(images,prep,road,world,OPTICAL_FROM_CARLA)
                attempts.append(dict(frame_id=int(world.get_snapshot().frame),fixture=_finite_json(fixture)))
                if all(fixture[c].get("qualified") for c in ("narrow","wide")):
                    break
            write_json(output/"depth-fixture-attempts.json",attempts)
            write_json(output/"depth-fixture.json",_finite_json(fixture))
            if not all(fixture[c].get("qualified") for c in ("narrow","wide")):
                raise RuntimeError("bounded pre-motion depth qualification did not pass both cameras")
            writers = {camera:archive.FrameArchiveWriter(output/f"packed_{camera}.npz", dtype=np.uint8,
                frame_shape=(6,128,256), expected_count=ticks) for camera in ("narrow", "wide")}
            with (output/"frames.jsonl").open("w",encoding="utf-8") as frames_file, (output/"evidence-frames.jsonl").open("w",encoding="utf-8") as evidence_file:
                for index in range(ticks):
                    world.tick()
                    snapshot = world.get_snapshot()
                    images, stamps = rig.read(snapshot)
                    packed = prep.pack_rgb(images["narrow_rgb"], images["wide_rgb"])
                    row = state(ego,snapshot)
                    yaw = row["rotation_world"][2]
                    lateral_accel = float(np.array([-math.sin(yaw),math.cos(yaw)])@np.array(row["acceleration_world"][:2]))
                    current_s,_ = road.project(row["position_world"][:2])
                    current = VehicleState(row["position_world"][0],row["position_world"][1],yaw,row["speed"],
                        row["angular_velocity_world"][2], timestamp_s=row["timestamp"], lateral_acceleration_mps2=lateral_accel)
                    command = teacher.step(current)
                    for camera,value in zip(("narrow", "wide"),packed): writers[camera].write(index,value)
                    metadata = dict(index=index,frame_id=row["frame_id"],timestamp=row["timestamp"],
                        desire=[1.,0.,0.,0.,0.,0.,0.,0.],traffic_convention=[1.,0.],action_t=[calibration.delay_s+.05,.35])
                    frames_file.write(json.dumps(metadata)+"\n")
                    row.update(index=index,route_s=current_s,lateral_acceleration_mps2=lateral_accel)
                    rows.append(row)
                    commands.append(asdict(command))
                    if index%5==0:
                        scene=controlled_scene_record(world,snapshot,ego,obstacle)
                        record = save_evidence_frame(output,index,snapshot,images,stamps,packed,prep,ego,obstacle,
                            controlled_scene=scene)
                        evidence_file.write(json.dumps(record)+"\n"); evidence_file.flush()
                        evidence.append(record)
                        edge_frame = dict(frame_id=row["frame_id"],timestamp_s=row["timestamp"],index=index,cameras={})
                        for camera,profile in zip(("narrow","wide"),prep.profiles):
                            if fixture[camera].get("qualified"):
                                luma = split_nv12(*image_to_nv12(images[camera+"_rgb"],"RGB"))[0]
                                edge_frame["cameras"][camera] = measure_camera(reference=road,current_s=current_s,
                                    tags=images[camera+"_semantic_segmentation"], depth=images[camera+"_depth"],
                                    camera_from_world=OPTICAL_FROM_CARLA@images[camera+"_rgb_camera_from_world"],
                                    intrinsics=profile.K,depth_mode=fixture[camera]["depth_mode"],road_line_tag=24,
                                    luma=luma,road_tag=1,exterior_tags=[2,10,25],pixel_center_offset=.5)
                        edge_frames.append(edge_frame)
                    batch_control(client,ego,throttle=command.throttle,brake=command.brake,steer=command.steer)
                    if index%20==0:
                        write_json(output/"progress.json",dict(event="unmarked_physical_capture",tick=index,ticks=ticks,
                            speed=row["speed"],route_s=current_s,output=str(output),accepted_labels=0))
                        print(json.dumps(dict(event="unmarked_physical_capture",tick=index,ticks=ticks,
                            speed=row["speed"],route_s=current_s,output=str(output),command=asdict(command)),default=str),flush=True)
                    if collisions:
                        raise RuntimeError("actual collision during physical behavior capture")
                    if not command.accepted:
                        raise RuntimeError("physical teacher rejected: "+", ".join(command.reasons))
            for writer in writers.values(): writer.finalize(complete=True)
            report["status"] = "captured_pending_observed_admission"
            report["archives"] = {camera:dict(path=f"packed_{camera}.npz",sha256=sha256_file(output/f"packed_{camera}.npz")) for camera in writers}
        except BaseException as error:
            report["status"] = "capture_failed"
            report["errors"].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        finally:
            for camera,writer in writers.items():
                try: writer.close()
                except Exception as error: report["cleanup_errors"].append("archive "+camera+": "+str(error))
            if rig is not None:
                try: rig.close()
                except Exception as error: report["cleanup_errors"].append("camera rig: "+str(error))
            for actor in reversed(actors):
                try: destroy_probe_actor(actor,world)
                except Exception as error: report["cleanup_errors"].append(str(error))
            try: apply_checked_settings(world or previous,original)
            except RuntimeError as error: report["cleanup_errors"].append(str(error))
            report["completed_ticks"] = len(rows)
            write_json(output/"telemetry.json",_finite_json(rows))
            write_json(output/"commands.json",_finite_json(commands))
            write_json(output/"collisions.json",collisions)
            write_json(output/"road-edge-samples.json",_finite_json(edge_frames))
            report["road_edges"] = summarize_frames(edge_frames) if edge_frames else None
            if target is not None and target.feasible and len(rows)>=3:
                try:
                    report["physical"] = qualify_teacher_rollout(target=target,target_sha256=target.contract_sha256,
                        timestamps_s=[r["timestamp"] for r in rows],positions_xyz=[r["position_world"] for r in rows],
                        yaw_rad=[r["rotation_world"][2] for r in rows],velocity_world=[r["velocity_world"] for r in rows],
                        lateral_acceleration_mps2=[r["lateral_acceleration_mps2"] for r in rows],collision_count=len(collisions))
                    if evidence and evidence[0]["obstacle"] is not None:
                        actors_seen = [item["obstacle"] for item in evidence]
                        positions = np.array([a["position_world_m"] for a in actors_seen])
                        velocity = np.array([a["velocity_world_mps"] for a in actors_seen])
                        yaw = np.unwrap([a["yaw_rad"] for a in actors_seen])
                        stationary = bool(np.max(np.linalg.norm(positions-positions[0],axis=1))<=.03
                            and np.max(np.linalg.norm(velocity,axis=1))<=.05 and np.max(np.abs(yaw-yaw[0]))<=math.radians(.5)
                            and len({a["actor_id"] for a in actors_seen})==1)
                        report["physical"]["static_actor_qa"] = dict(passed=stationary,observed_states=len(actors_seen),
                            scope="same-frame simulator QA, not visual evidence")
                        if not stationary:
                            report["physical"].update(passed=False,physical_passed=False)
                            report["physical"]["reasons"].append("stationary_teacher_obstacle_changed")
                except Exception as error: report["physical"] = dict(passed=False,error=str(error))
            report["files"] = {p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob("*") if p.is_file()}
            write_json(output/"capture-report.json",_finite_json(report))
    return report,output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--case",choices=("clear_road","parked_car","roadside_pedestrian"),required=True)
    parser.add_argument("--ticks",type=int,default=1200)
    parser.add_argument("--start-s",type=float,default=10.)
    parser.add_argument("--paused-supervisor-pid",type=int)
    args = parser.parse_args()
    report, output = run_capture(**vars(args))
    print(json.dumps(dict(status=report["status"],output=str(output),physical=report.get("physical"),errors=report["errors"])),flush=True)
    return 0 if report["status"] == "captured_pending_observed_admission" and report.get("physical",{}).get("passed") and not report["cleanup_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
