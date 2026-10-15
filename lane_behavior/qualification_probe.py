"""Bounded stationary rendered qualification, never a training-data publisher.

Requires an explicit idle CARLA port and obtains the collection world's OS lock
before connecting. Active collection ports and worlds with existing actors are
refused. The generated world remains loaded after cleanup; its prior physics
settings are restored. No invocation occurs on import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import traceback
import uuid

import numpy as np

from lane_dataset.config import Config, sha256_file, write_json
from lane_dataset.depth_fixture import collision_depth_fixture
from lane_dataset.geometry import LaneReference
from lane_dataset.preprocessing import StockPreprocessor, image_to_nv12, split_nv12
from lane_dataset.render_quality import apply_production_weather
from lane_dataset.rendered_boundaries import measure_camera, summarize_frames
from lane_dataset.simulator import (CameraRig, ExactSensorQueue, apply_checked_settings,
    batch_control, configure_world, connect, spawn_vehicle, verify_camera_attributes, world_owner)
from lane_dataset.traffic import convex_hull
from lane_dataset.visibility import decode_carla_depth_bgra
from .evidence import (ActorState, FrameStamp, Observation, build_static_proof,
                       decode_carla_instance_bgra)
from .scenarios import CASES


OPTICAL_FROM_CARLA = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]], float)
INSTANCE_ENCODING_SOURCE = ("https://raw.githubusercontent.com/carla-simulator/carla/0.9.16/"
    "Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Game/Tagger.cpp")


def load_probe_task(plan_path, case):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("status") != "offline_candidate_only" or plan.get("split") != "development"
            or plan.get("rendered_qualification") is not False):
        raise ValueError("probe requires an unqualified development candidate plan")
    matches = [task for task in plan.get("tasks", []) if task.get("unmarked_behavior", {}).get("case") == case]
    if len(matches) != 1:
        raise ValueError("plan must contain exactly one requested behavior case")
    task = matches[0]
    if (task.get("unmarked_road") is not True or task.get("training_ready") is not False
            or task.get("capture_contract") != "unmarked_rht_obstacle_v1"):
        raise ValueError("unsupported unmarked behavior capture contract")
    xodr_path = Path(task["opendrive_path"])
    if sha256_file(xodr_path) != task["route"]["source_xodr_sha256"]:
        raise ValueError("candidate OpenDRIVE source hash mismatch")
    return task


def refuse_active_collection_port(root, port, paused_supervisor_pid=None):
    """Protect slots between worker jobs, when their world lock can be free."""
    import psutil
    status_path = Path(root)/"reports/automatic/status.json"
    if not status_path.exists():
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    ports = {int(slot["port"]) for slot in status.get("configured_workers", [])}
    pid = status.get("pid")
    if port in ports and status.get("status") == "running" and isinstance(pid, int) and psutil.pid_exists(pid):
        if paused_supervisor_pid != pid or psutil.Process(pid).status() != psutil.STATUS_STOPPED:
            raise RuntimeError("requested port belongs to the running automatic collector; explicit matching suspended supervisor required")
    elif paused_supervisor_pid is not None:
        raise RuntimeError("explicit suspended supervisor must match a live configured collector")


def probe_route_s(task, explicit=None):
    route_length = float(task["route"]["length_m"])
    obstacles = task["unmarked_behavior"]["obstacles"]
    value = (float(obstacles[0]["route_s_m"])-20. if obstacles else 70.) if explicit is None else float(explicit)
    if not math.isfinite(value) or not 10. <= value <= route_length-40.:
        raise ValueError("probe pose must retain at least 10 m behind and 40 m forward road")
    return value


def actor_document(actor, snapshot):
    """Use the same frozen synchronous snapshot as the sensor frame."""
    item = snapshot.find(actor.id)
    if item is None:
        raise RuntimeError("spawned actor missing from captured world snapshot")
    transform = item.get_transform()
    velocity, angular = item.get_velocity(), item.get_angular_velocity()
    vertices = actor.bounding_box.get_world_vertices(transform)
    box = actor.bounding_box
    return dict(actor_id=int(actor.id), blueprint=actor.type_id,
        transform_world=np.asarray(transform.get_matrix()).tolist(),
        position_world_m=[transform.location.x, transform.location.y, transform.location.z],
        yaw_rad=math.radians(transform.rotation.yaw),
        velocity_world_mps=[velocity.x, velocity.y, velocity.z], yaw_rate_rps=math.radians(angular.z),
        bounding_box=dict(extent_m=[box.extent.x, box.extent.y, box.extent.z],
            center_actor_m=[box.location.x, box.location.y, box.location.z],
            rotation_degrees=[box.rotation.roll, box.rotation.pitch, box.rotation.yaw],
            vertices_world_xyz=[[point.x,point.y,point.z] for point in vertices]),
        source="simulator_actor_state_qa", entire_footprint_observed=False)


class ProofCameraRig(CameraRig):
    """Existing optical rig plus aligned raw instance data and frame stamps."""
    def __init__(self, world, ego):
        import carla
        super().__init__(world, ego, semantic_qa=True)
        try:
            for name in ("narrow", "wide"):
                key = name+"_instance_segmentation"
                bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
                attrs = dict(self.specs[name+"_depth"]["attributes"])
                for attribute, value in attrs.items():
                    bp.set_attribute(attribute, value)
                actor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.5,z=1.3)), attach_to=ego)
                self.actors[key] = actor
                verify_camera_attributes(attrs, actor.attributes)
                queue = ExactSensorQueue(key)
                self.queues[key] = queue
                self.specs[key] = dict(attributes=attrs, actor_attributes=dict(actor.attributes),
                    focal_px=self.specs[name+"_rgb"]["focal_px"], mount=self.specs[name+"_rgb"]["mount"])
                actor.listen(queue.callback)
        except BaseException:
            self.close()
            raise

    def read(self, snapshot):
        images, stamps = {}, {}
        for key, queue in self.queues.items():
            image = queue.get(snapshot.frame, snapshot.timestamp.elapsed_seconds)
            if (image.width != self.width or image.height != self.height
                    or not math.isclose(image.fov,float(self.specs[key]["attributes"]["fov"]),rel_tol=1e-6)):
                raise RuntimeError("proof sensor dimensions/FOV differ from recorded optical model")
            raw = np.frombuffer(image.raw_data, np.uint8).reshape(self.height,self.width,4).copy()
            images[key+"_bgra"] = raw
            images[key+"_camera_from_world"] = np.asarray(image.transform.get_inverse_matrix())
            stamps[key] = dict(frame_id=int(image.frame), timestamp_s=float(image.timestamp), camera=key.split("_",1)[0])
            if key.endswith("_rgb"):
                images[key] = raw[...,2::-1].copy()
            elif key.endswith("_depth"):
                images[key] = decode_carla_depth_bgra(raw).astype(np.float32)
            elif key.endswith("_instance_segmentation"):
                semantic, instance = decode_carla_instance_bgra(raw)
                images[key], images[key+"_semantic"] = instance, semantic
            else:
                images[key] = raw[...,2].copy()
        for camera in ("narrow", "wide"):
            base = images[camera+"_rgb_camera_from_world"]
            for modality in ("depth", "semantic_segmentation", "instance_segmentation"):
                if not np.allclose(base,images[camera+"_"+modality+"_camera_from_world"],atol=1e-7,rtol=0):
                    raise RuntimeError("proof camera modalities do not share an exact transform")
        return images, stamps


def _spawn_obstacle(world, task, reference, actors):
    """Measure and place one static actor before starting any camera sensor.

    Placement is completed before sensors begin. This render-only fixture is
    explicitly kinematic and cannot qualify a physics-driven training capture.
    """
    import carla
    specs = task["unmarked_behavior"]["obstacles"]
    if not specs:
        return None, None
    spec = specs[0]
    s = float(spec["route_s_m"])
    center, yaw, _, _ = reference.sample(s)
    right = np.array([-math.sin(float(yaw)), math.cos(float(yaw)), 0.])
    edge = reference._right(s)
    candidates = sorted(world.get_blueprint_library().filter(spec["blueprint_filter"]), key=lambda bp:bp.id)
    if not candidates:
        raise RuntimeError("requested obstacle blueprint is unavailable")
    bp = candidates[0]
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "unmarked_render_qualification_only")
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "true")
    tentative = edge + right*.5 + np.array([0.,0.,1.])
    transform = carla.Transform(carla.Location(*map(float,tentative)),carla.Rotation(yaw=math.degrees(float(yaw))))
    actor = world.spawn_actor(bp, transform)
    actors.append(actor)
    actor.set_simulate_physics(False)
    # CARLA's just-spawned client actor can still return the previous snapshot's
    # default origin. Advance ownership once before measuring its actual pose.
    world.tick()
    actual_transform = actor.get_transform()
    actual_origin = np.array([actual_transform.location.x,actual_transform.location.y,actual_transform.location.z])
    vertices = np.array([[v.x,v.y,v.z] for v in actor.bounding_box.get_world_vertices(actual_transform)])
    inward = float(np.min((vertices-edge)@right))
    lateral_delta = -float(spec["maximum_road_intrusion_m"])-inward
    target = actual_origin+right*lateral_delta
    ground_labels={carla.CityObjectLabel.Roads,carla.CityObjectLabel.Sidewalks,
                   carla.CityObjectLabel.Ground,carla.CityObjectLabel.Terrain}
    hits=world.cast_ray(carla.Location(float(target[0]),float(target[1]),float(edge[2])+5.),
                       carla.Location(float(target[0]),float(target[1]),float(edge[2])-5.))
    ground=[hit for hit in hits if hit.label in ground_labels]
    if not ground:
        raise RuntimeError("no independently collision-surveyed ground under obstacle placement")
    ground_z=max(float(hit.location.z) for hit in ground)
    target[2] += ground_z-float(vertices[:,2].min())
    actor.set_transform(carla.Transform(carla.Location(*map(float,target)),transform.rotation))
    world.tick()
    actual_vertices=np.array([[v.x,v.y,v.z] for v in actor.bounding_box.get_world_vertices(actor.get_transform())])
    actual_intrusion=-float(np.min((actual_vertices-edge)@right))
    if (abs(actual_intrusion-float(spec["maximum_road_intrusion_m"]))>.02
            or abs(float(actual_vertices[:,2].min())-ground_z)>.02):
        raise RuntimeError("obstacle placement readback disagrees with measured road intrusion or ground contact")
    return actor, spec


def destroy_probe_actor(actor, world):
    """Confirm removal rather than interpreting a void/false return as failure."""
    if world.get_actor(actor.id) is None:
        return
    try:
        result = actor.destroy()
    except RuntimeError:
        if world.get_actor(actor.id) is None:
            return
        raise
    if result is not True:
        # Refresh the synchronous episode snapshot after a void/ambiguous API
        # return; no other owner can tick this world while the probe lock holds.
        world.tick()
        if world.get_actor(actor.id) is not None:
            raise RuntimeError(f"actor_destroy_failed:{actor.id}")


def _proofs_from_saved(output, frames, fixture, actor, category):
    results = {}
    for camera in ("narrow", "wide"):
        observations, states = [], []
        selected = [frames[-11],frames[-6],frames[-1]]
        for frame in selected:
            record = frame["cameras"][camera]
            stamps = {key:FrameStamp(**value) for key,value in record["modality_stamps"].items()}
            stamp = stamps["rgb"]
            with np.load(output/record["payload"],allow_pickle=False) as arrays:
                observations.append(Observation(stamp=stamp,modality_stamps=stamps,
                    **{name:arrays[name].copy() for name in ("rgb","depth_m","semantic_ids","instance_ids",
                        "packed","intrinsics","camera_from_world","model_to_source","instance_bgra")},
                    scene_actor_ids=tuple(frame["scene_actor_ids"]),
                    depth_mode=fixture[camera].get("depth_mode") or "unqualified",
                    depth_fixture_sha256=sha256_file(output/"depth-fixture.json"),
                    depth_qualified=fixture[camera].get("qualified") is True))
            document = frame["obstacle"]
            states.append(ActorState(stamp=stamp, **{key:document[key] for key in
                ("actor_id","position_world_m","yaw_rad","velocity_world_mps","yaw_rate_rps")}))
        polygon = convex_hull(np.array(selected[-1]["obstacle"]["bounding_box"]["vertices_world_xyz"])[:,:2])
        results[camera] = build_static_proof(observations,states,actor_id=actor.id,instance_id=actor.id,
            category=category,conservative_footprint_world_xy=polygon,decision=observations[-1].stamp)
    return results


def run_probe(*, port, root, plan, case, frames=24, ego_route_s=None, paused_supervisor_pid=None):
    """Run one explicit, bounded stationary probe; never create accepted labels."""
    import carla
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("explicit valid CARLA port required")
    if type(frames) is not int or not 11 <= frames <= 60:
        raise ValueError("probe frames must be between 11 and 60")
    root = Path(root).resolve()
    task = load_probe_task(plan,case)
    current_s = probe_route_s(task,ego_route_s)
    refuse_active_collection_port(root,port,paused_supervisor_pid)
    output = root/"reports/unmarked/qualification"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8])
    report = dict(schema_version=1,scope="stationary_rendered_qualification_only",training_ready=False,
        accepted_labels=0,physics_driving_qualified=False,case=case,episode_id=task["episode_id"],
        port=port,plan=str(Path(plan).resolve()),plan_sha256=sha256_file(plan),
        instance_encoding_source=INSTANCE_ENCODING_SOURCE,source_contract="carla-0.9.16_G_plus_256_B",
        requested_frames=frames,frame_records=[],errors=[],cleanup_errors=[])
    # capture_development takes world_owner(port, root), not root/locks.
    # Using another directory would create an independent, ineffective lock.
    with world_owner(port,root):
        refuse_active_collection_port(root,port,paused_supervisor_pid)
        client = connect(port)
        previous = client.get_world()
        present = [a for a in previous.get_actors() if a.type_id.startswith(("vehicle.","walker.","sensor."))]
        if present:
            raise RuntimeError("refusing to replace a world containing existing vehicle, walker, or sensor actors")
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"candidate-task.json",task)
        report["candidate_task_sha256"] = sha256_file(output/"candidate-task.json")
        report["source_sha256"] = {name:sha256_file(Path(__file__).with_name(name))
            for name in ("qualification_probe.py","scenarios.py","evidence.py")}
        report["stationary_fixture_policy"] = "ego_settled_under_brake_then_frozen_obstacle_kinematic_no_training_publication"
        report["stationary_fixture_overrides"] = dict(initial_obstacle_placement_teleport_before_sensors=True,
            obstacle_teleport_during_recording=False,qualifies_production_physics=False)
        old_settings = previous.get_settings()
        keys = ("synchronous_mode","fixed_delta_seconds","substepping","max_substep_delta_time","max_substeps","no_rendering_mode")
        original = {key:getattr(old_settings,key) for key in keys}
        actors,rig,world = [],None,None
        try:
            config = Config()
            expected = dict(synchronous_mode=True,fixed_delta_seconds=config.dt,substepping=True,
                max_substep_delta_time=config.max_substep_delta_time,max_substeps=config.max_substeps,no_rendering_mode=False)
            apply_checked_settings(previous,expected)
            if task.get("native_unmarked") is True and task.get("procedural") is False:
                world,_ = configure_world(client,task["town"],render=True)
                report["loaded_source_kind"] = "native_map"
            else:
                world = client.generate_opendrive_world(Path(task["opendrive_path"]).read_text(encoding="utf-8"),
                    carla.OpendriveGenerationParameters(vertex_distance=.5,max_road_length=50.,wall_height=0.,
                        additional_width=0.,smooth_junctions=True,enable_mesh_visibility=True))
                report["loaded_source_kind"] = "generated_opendrive_fixture"
            report["world_settings"] = apply_checked_settings(world,expected)
            report["render_profile"] = apply_production_weather(world,carla,0)
            route = task["route"]
            reference = LaneReference(route["left_xyz"],route["right_xyz"],route["arc_m"])
            roads = route["unmarked_road"]
            road_reference = LaneReference(roads["road_left_edge_xyz"],roads["road_right_edge_xyz"],route["arc_m"])
            xyz = reference.sample(current_s)[0]
            waypoint = world.get_map().get_waypoint(carla.Location(*map(float,xyz)))
            if waypoint is None or waypoint.lane_id != task["start"]["lane_id"]:
                raise RuntimeError("probe ego pose did not resolve to the planned right-hand topology lane")
            wp_location,wp_right=waypoint.transform.location,waypoint.transform.get_right_vector()
            ego_offset=float(np.dot(xyz-np.array([wp_location.x,wp_location.y,wp_location.z]),
                np.array([wp_right.x,wp_right.y,wp_right.z])))
            ego = spawn_vehicle(world,waypoint,offset=ego_offset)
            actors.append(ego)
            batch_control(client,ego,brake=1.)
            for _ in range(30):
                world.tick()
            ego.set_simulate_physics(False)
            obstacle,spec = _spawn_obstacle(world,task,reference,actors)
            rig = ProofCameraRig(world,ego)
            prep = StockPreprocessor()
            report["camera_specs"] = rig.specs
            for _ in range(24):
                world.tick()
                rig.read(world.get_snapshot())
            fixture,edge_frames = None,[]
            for index in range(frames):
                world.tick()
                snapshot = world.get_snapshot()
                images,stamps = rig.read(snapshot)
                if fixture is None:
                    fixture = collision_depth_fixture(images,prep,road_reference,world,OPTICAL_FROM_CARLA)
                    write_json(output/"depth-fixture.json",fixture)
                packed = prep.pack_rgb(images["narrow_rgb"],images["wide_rgb"])
                record = dict(index=index,frame_id=int(snapshot.frame),timestamp_s=float(snapshot.timestamp.elapsed_seconds),
                    scene_actor_ids=sorted(int(item.id) for item in snapshot),ego=actor_document(ego,snapshot),
                    obstacle=actor_document(obstacle,snapshot) if obstacle else None,cameras={})
                edge_frame = dict(frame_id=int(snapshot.frame),cameras={})
                for camera_index,(camera,profile,warp) in enumerate(zip(("narrow","wide"),prep.profiles,prep.warps)):
                    optical = OPTICAL_FROM_CARLA@images[camera+"_rgb_camera_from_world"]
                    path = output/f"frame-{index:03d}-{camera}.npz"
                    np.savez_compressed(path,rgb=images[camera+"_rgb"],depth_m=images[camera+"_depth"],
                        semantic_ids=images[camera+"_instance_segmentation_semantic"],
                        instance_ids=images[camera+"_instance_segmentation"],
                        instance_bgra=images[camera+"_instance_segmentation_bgra"],
                        depth_bgra=images[camera+"_depth_bgra"],semantic_camera_ids=images[camera+"_semantic_segmentation"],
                        rgb_bgra=images[camera+"_rgb_bgra"],packed=packed[camera_index],
                        intrinsics=profile.K,camera_from_world=optical,model_to_source=warp)
                    record["cameras"][camera] = dict(payload=path.name,payload_sha256=sha256_file(path),
                        modality_stamps={"rgb":stamps[camera+"_rgb"],"packed":stamps[camera+"_rgb"],
                            "depth":stamps[camera+"_depth"],"semantic":stamps[camera+"_instance_segmentation"],
                            "instance":stamps[camera+"_instance_segmentation"]},
                        separate_semantic_sensor_stamp=stamps[camera+"_semantic_segmentation"])
                    if fixture[camera].get("qualified"):
                        nv12,layout = image_to_nv12(images[camera+"_rgb"],"RGB")
                        luma = split_nv12(nv12,layout)[0]
                        edge_frame["cameras"][camera] = measure_camera(reference=road_reference,current_s=current_s,
                            tags=images[camera+"_semantic_segmentation"],depth=images[camera+"_depth"],
                            camera_from_world=optical,intrinsics=profile.K,depth_mode=fixture[camera]["depth_mode"],
                            road_line_tag=int(carla.CityObjectLabel.RoadLines),luma=luma,road_tag=int(carla.CityObjectLabel.Roads),
                            exterior_tags=[int(carla.CityObjectLabel.Sidewalks),int(carla.CityObjectLabel.Terrain),int(carla.CityObjectLabel.Ground)],
                            pixel_center_offset=.5)
                report["frame_records"].append(record)
                edge_frames.append(edge_frame)
                print(json.dumps(dict(event="unmarked_render_probe",frame=index+1,total=frames,output=str(output))),flush=True)
            write_json(output/"road-edge-samples.json",edge_frames)
            report["road_edge_samples_sha256"] = sha256_file(output/"road-edge-samples.json")
            report["depth_fixture_sha256"] = sha256_file(output/"depth-fixture.json")
            report["road_edges"] = summarize_frames(edge_frames)
            report["depth_fixture"] = fixture
            if obstacle is not None:
                report["obstacle_proofs"] = _proofs_from_saved(output,report["frame_records"],fixture,obstacle,spec["kind"])
            report["status"] = "stationary_render_probe_complete"
            report["qualified_for_training"] = False
            report["remaining"] = ["physics_driven_avoidance_and_recovery_capture", "independent_behavior_admission",
                "model_training_loader_and_objective_integration", "split_and_release_qualification"]
        except BaseException as error:
            report["status"] = "probe_failed"
            report["errors"].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        finally:
            if rig is not None:
                rig.close()
            for actor in reversed(actors):
                try:
                    destroy_probe_actor(actor,world)
                except RuntimeError as error:
                    report["cleanup_errors"].append(str(error))
            try:
                apply_checked_settings(world or previous,original)
                report["original_physics_settings_restored"] = True
            except RuntimeError as error:
                report["cleanup_errors"].append(str(error))
            write_json(output/"probe-report.json",report)
    return report,output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--case",choices=CASES,required=True)
    parser.add_argument("--frames",type=int,default=24)
    parser.add_argument("--ego-route-s",type=float)
    parser.add_argument("--paused-supervisor-pid",type=int,
        help="explicit matching automatic supervisor PID, which must already be suspended")
    args = parser.parse_args()
    report,output = run_probe(port=args.port,root=args.root,plan=args.plan,case=args.case,
        frames=args.frames,ego_route_s=args.ego_route_s,paused_supervisor_pid=args.paused_supervisor_pid)
    print(json.dumps(dict(status=report["status"],report=str(output/"probe-report.json"),
        accepted_labels=0,training_ready=False),indent=2))
    return 0 if report["status"] == "stationary_render_probe_complete" and not report["cleanup_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
