"""Isolated CARLA ownership and exact synchronous sensor delivery."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import queue
import time

import numpy as np

from .config import Config, write_json


@contextmanager
def world_owner(port, lock_root):
    """OS file lock is released on crash; never silently steal a live tick owner."""
    path = Path(lock_root) / f"world-{port}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def connect(port, timeout=40):
    import carla
    client = carla.Client("127.0.0.1", port)
    client.set_timeout(timeout)
    if client.get_server_version() != "0.9.16":
        raise RuntimeError("CARLA server version mismatch")
    if client.get_client_version() != "0.9.16":
        raise RuntimeError("CARLA client version mismatch; use the project Python 3.12 environment")
    return client


def apply_checked_settings(world, expected):
    settings=world.get_settings()
    for name,value in expected.items():
        setattr(settings,name,value)
    world.apply_settings(settings)
    returned=world.get_settings()
    actual={name:getattr(returned,name) for name in expected}
    for name,value in expected.items():
        if isinstance(value,float):
            if actual[name] is None or not math.isclose(actual[name],value,rel_tol=1e-6,abs_tol=1e-8):
                raise RuntimeError(f"physics readback mismatch: {name}={actual[name]}")
        elif actual[name]!=value:
            raise RuntimeError(f"physics readback mismatch: {name}={actual[name]}")
    return actual


def configure_world(client, town, render=True):
    config = Config()
    old = client.get_world()
    settings = old.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = config.dt
    settings.substepping = True
    settings.max_substep_delta_time = config.max_substep_delta_time
    settings.max_substeps = config.max_substeps
    settings.no_rendering_mode = not render
    old.apply_settings(settings)
    # Preserve synchronous settings across reload, as required for determinism.
    world = client.load_world(town, False)
    returned = world.get_settings()
    expected = dict(synchronous_mode=True, fixed_delta_seconds=config.dt, substepping=True,
                    max_substep_delta_time=config.max_substep_delta_time,
                    max_substeps=config.max_substeps, no_rendering_mode=not render)
    actual = {k: getattr(returned, k) for k in expected}
    for name, value in expected.items():
        if isinstance(value, float):
            if not math.isclose(actual[name], value, rel_tol=1e-6, abs_tol=1e-8):
                raise RuntimeError(f"physics readback mismatch: {name}={actual[name]}")
        elif actual[name] != value:
            raise RuntimeError(f"physics readback mismatch: {name}")
    return world, actual


def batch_control(client, ego, throttle=0, brake=0, steer=0):
    import carla
    control = carla.VehicleControl(throttle=float(np.clip(throttle, 0, 1)),
                                   brake=float(np.clip(brake, 0, 1)),
                                   steer=float(np.clip(steer, -1, 1)))
    result = client.apply_batch_sync([carla.command.ApplyVehicleControl(ego.id, control)], False)
    if len(result) != 1 or result[0].has_error():
        raise RuntimeError(f"vehicle control batch failed: {result}")


def spawn_vehicle(world, waypoint, offset=0, heading_error=0):
    import carla
    transform = waypoint.transform
    right = transform.get_right_vector()
    transform.location += carla.Location(x=right.x*offset, y=right.y*offset, z=0.25)
    transform.rotation.yaw += heading_error
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "lane_dataset_ego")
    ego = world.spawn_actor(bp, transform)
    return ego


def state(ego, snapshot):
    transform, velocity, accel, angular = ego.get_transform(), ego.get_velocity(), ego.get_acceleration(), ego.get_angular_velocity()
    control=ego.get_control()
    return dict(frame_id=int(snapshot.frame), timestamp=float(snapshot.timestamp.elapsed_seconds),
                position_world=[transform.location.x,transform.location.y,transform.location.z],
                rotation_world=np.deg2rad([transform.rotation.roll,transform.rotation.pitch,transform.rotation.yaw]).tolist(),
                velocity_world=[velocity.x,velocity.y,velocity.z],
                acceleration_world=[accel.x,accel.y,accel.z],
                angular_velocity_world=np.deg2rad([angular.x,angular.y,angular.z]).tolist(),
                speed=float(math.hypot(velocity.x,velocity.y)), steer=float(control.steer),
                throttle=float(control.throttle),brake=float(control.brake),gear=int(control.gear),
                hand_brake=bool(control.hand_brake),reverse=bool(control.reverse),manual_gear_shift=bool(control.manual_gear_shift))


class ExactSensorQueue:
    def __init__(self, name):
        self.name = name
        self.queue = queue.Queue(maxsize=8)
        self.last_frame = None
        self.overflow = False

    def callback(self, image):
        try:
            self.queue.put_nowait(image)
        except queue.Full:
            self.overflow = True

    def get(self, frame, timestamp, timeout=30):
        if self.overflow:
            raise RuntimeError(f"{self.name}: sensor queue overflow")
        image = self.queue.get(timeout=timeout)
        if image.frame != frame:
            raise RuntimeError(f"{self.name}: missing/duplicate/out-of-order frame expected {frame}, received {image.frame}")
        if self.last_frame is not None and image.frame != self.last_frame + 1:
            raise RuntimeError(f"{self.name}: non-consecutive sensor sequence")
        if abs(image.timestamp - timestamp) > 1e-6:
            raise RuntimeError(f"{self.name}: sensor timestamp mismatch")
        self.last_frame = image.frame
        return image


def verify_camera_attributes(expected, actual):
    """Reject a sensor whose server attributes differ from the recorded model."""
    for name,value in expected.items():
        observed=actual.get(name)
        if str(value).lower() in ("true","false"):
            equal=str(observed).lower()==str(value).lower()
        else:
            try:
                equal=math.isclose(float(observed),float(value),rel_tol=1e-6,abs_tol=1e-6)
            except (TypeError,ValueError):
                equal=observed==value
        if not equal:
            raise RuntimeError(f"camera attribute readback mismatch: {name}={observed}, expected {value}")


class CameraRig:
    width, height = 1928, 1208
    def __init__(self, world, ego, *, semantic_qa=False):
        import carla
        self.actors, self.queues, self.specs = {}, {}, {}
        mount = carla.Transform(carla.Location(x=1.5, z=1.3))
        try:
            for name, focal in (("narrow",2648.),("wide",567.)):
                for kind in (("rgb", "depth", "semantic_segmentation") if semantic_qa else ("rgb", "depth")):
                    key = name + "_" + kind
                    bp = world.get_blueprint_library().find("sensor.camera." + kind)
                    attrs = dict(image_size_x=str(self.width),image_size_y=str(self.height),
                                 fov=str(math.degrees(2*math.atan(self.width/(2*focal)))),sensor_tick="0.0",
                                 lens_k="0.0",lens_kcube="0.0",lens_circle_multiplier="0.0")
                    if bp.has_attribute("enable_postprocess_effects"):
                        # RGB needs CARLA's tone mapping/exposure, otherwise HDR
                        # skies clip to white. Depth remains a metric encoding.
                        attrs["enable_postprocess_effects"] = "true" if kind=="rgb" else "false"
                    if kind=="rgb":
                        # Instantaneous pinhole projection is the explicitly
                        # assumed camera model used by depth/pixel evidence.
                        attrs["motion_blur_intensity"]="0.0"
                        attrs["chromatic_aberration_intensity"]="0.0"
                    for key_attr, value in attrs.items():
                        bp.set_attribute(key_attr,value)
                    actor = world.spawn_actor(bp,mount,attach_to=ego)
                    self.actors[key]=actor
                    verify_camera_attributes(attrs,actor.attributes)
                    q = ExactSensorQueue(key)
                    actor.listen(q.callback)
                    self.actors[key],self.queues[key] = actor,q
                    self.specs[key] = dict(attributes=attrs,actor_attributes=dict(actor.attributes),
                        projection_assumption="pinhole_no_lens_distortion_instantaneous_exposure",
                        focal_px=focal,mount=dict(x=1.5,y=0,z=1.3,roll=0,pitch=0,yaw=0))
        except BaseException:
            self.close()
            raise

    def read(self, snapshot):
        result = {}
        for key,q in self.queues.items():
            image = q.get(snapshot.frame,snapshot.timestamp.elapsed_seconds)
            if (image.width!=self.width or image.height!=self.height or
                not math.isclose(image.fov,float(self.specs[key]["attributes"]["fov"]),rel_tol=1e-6,abs_tol=1e-6)):
                raise RuntimeError(f"{key}: returned image dimensions/FOV differ from the camera model")
            bgra = np.frombuffer(image.raw_data,np.uint8).reshape(self.height,self.width,4)
            if key.endswith("rgb"):
                result[key] = bgra[...,2::-1].copy()
            elif key.endswith("semantic_segmentation"):
                result[key] = bgra[...,2].copy()
            else:
                result[key] = (bgra[...,2].astype(np.uint32) + 256*bgra[...,1].astype(np.uint32) + 65536*bgra[...,0].astype(np.uint32)).astype(np.float32) * (1000/16777215)
            result[key+"_camera_from_world"] = np.asarray(image.transform.get_inverse_matrix())
        for camera in ("narrow","wide"):
            if not np.allclose(result[camera+"_rgb_camera_from_world"],result[camera+"_depth_camera_from_world"],rtol=0,atol=1e-7):
                raise RuntimeError(f"{camera}: RGB/depth transforms differ")
            semantic=camera+"_semantic_segmentation_camera_from_world"
            if semantic in result and not np.allclose(result[camera+"_rgb_camera_from_world"],result[semantic],rtol=0,atol=1e-7):
                raise RuntimeError(f"{camera}: RGB/semantic transforms differ")
        return result

    def close(self):
        for actor in self.actors.values():
            try:
                actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        self.actors.clear()
