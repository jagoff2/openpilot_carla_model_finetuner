import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from lane_behavior import qualification_probe as probe
from lane_dataset.config import sha256_file
from lane_dataset.simulator import ExactSensorQueue


def candidate(tmp_path):
    road = tmp_path/"road.xodr"
    road.write_text("<OpenDRIVE/>")
    task = dict(episode_id="probe-test",unmarked_road=True,training_ready=False,capture_contract="unmarked_rht_obstacle_v1",
        opendrive_path=str(road),route=dict(source_xodr_sha256=sha256_file(road),length_m=240.),
        unmarked_behavior=dict(case="parked_car",obstacles=[dict(route_s_m=100.)]))
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(dict(status="offline_candidate_only",split="development",
        rendered_qualification=False,tasks=[task])))
    return path,task,road


def test_probe_plan_requires_exact_case_and_hash_before_any_simulator_access(tmp_path):
    path,task,road = candidate(tmp_path)
    assert probe.load_probe_task(path,"parked_car") == task
    with pytest.raises(ValueError,match="exactly one"):
        probe.load_probe_task(path,"roadside_pedestrian")
    road.write_text("<changed/>")
    with pytest.raises(ValueError,match="hash mismatch"):
        probe.load_probe_task(path,"parked_car")


def test_probe_pose_retains_observation_envelope(tmp_path):
    _,task,_ = candidate(tmp_path)
    assert probe.probe_route_s(task) == 80.
    for value in (float("nan"),0.,220.):
        with pytest.raises(ValueError,match="40 m forward"):
            probe.probe_route_s(task,value)


def test_running_collector_slot_is_refused_even_between_jobs(tmp_path,monkeypatch):
    import psutil
    directory = tmp_path/"reports/automatic"
    directory.mkdir(parents=True)
    (directory/"status.json").write_text(json.dumps(dict(status="running",pid=os.getpid(),
        configured_workers=[dict(port=2400),dict(port=2500)],active_workers=[])))
    with pytest.raises(RuntimeError,match="running automatic collector"):
        probe.refuse_active_collection_port(tmp_path,2400)
    with pytest.raises(RuntimeError,match="suspended supervisor"):
        probe.refuse_active_collection_port(tmp_path,2400,os.getpid())
    monkeypatch.setattr(psutil,"Process",lambda pid:SimpleNamespace(status=lambda:psutil.STATUS_STOPPED))
    probe.refuse_active_collection_port(tmp_path,2400,os.getpid())
    with pytest.raises(RuntimeError,match="suspended supervisor"):
        probe.refuse_active_collection_port(tmp_path,2400,os.getpid()+1)
    probe.refuse_active_collection_port(tmp_path,2600)


def test_ownership_lock_precedes_client_connection(tmp_path,monkeypatch):
    pytest.importorskip("carla")
    path,_,_ = candidate(tmp_path)
    class HeldLock:
        def __enter__(self):
            raise OSError("world already owned")
        def __exit__(self,*args):
            pass
    def capture_world_lock(port,lock_root):
        assert port == 2600
        assert lock_root == tmp_path.resolve()  # Must match capture.py exactly.
        return HeldLock()
    monkeypatch.setattr(probe,"world_owner",capture_world_lock)
    monkeypatch.setattr(probe,"connect",lambda *args:pytest.fail("connected before obtaining ownership"))
    with pytest.raises(OSError,match="already owned"):
        probe.run_probe(port=2600,root=tmp_path,plan=path,case="parked_car")
    assert not (tmp_path/"reports/unmarked/qualification").exists()


def make_rig(*,wrong_transform=False):
    rig = object.__new__(probe.ProofCameraRig)
    rig.width,rig.height = 4,2
    rig.queues,rig.specs = {},{}
    for camera in ("narrow","wide"):
        for kind in ("rgb","depth","semantic_segmentation","instance_segmentation"):
            name = camera+"_"+kind
            queue = ExactSensorQueue(name)
            raw = np.zeros((2,4,4),np.uint8)
            raw[...,2] = 14
            raw[...,1] = 42
            raw[...,0] = 1
            transform = np.eye(4)
            if wrong_transform and name == "wide_instance_segmentation":
                transform[0,3] = 1.
            image = SimpleNamespace(frame=12,timestamp=.6,width=4,height=2,fov=90.,raw_data=raw.tobytes(),
                transform=SimpleNamespace(get_inverse_matrix=lambda matrix=transform:matrix))
            queue.callback(image)
            rig.queues[name] = queue
            rig.specs[name] = dict(attributes=dict(fov="90"))
    return rig


def test_proof_rig_preserves_exact_sensor_frames_raw_bytes_and_instance_identity():
    rig = make_rig()
    snapshot = SimpleNamespace(frame=12,timestamp=SimpleNamespace(elapsed_seconds=.6))
    images,stamps = rig.read(snapshot)
    assert set(stamps) == set(rig.queues)
    assert all(value["frame_id"] == 12 and value["timestamp_s"] == .6 for value in stamps.values())
    assert np.all(images["narrow_instance_segmentation"] == 298)
    assert np.all(images["narrow_instance_segmentation_semantic"] == 14)
    assert images["wide_depth_bgra"].dtype == np.uint8
    assert images["wide_depth"].dtype == np.float32
    np.testing.assert_array_equal(images["narrow_rgb"][0,0],[14,42,1])


def test_proof_rig_rejects_camera_transform_misalignment():
    rig = make_rig(wrong_transform=True)
    with pytest.raises(RuntimeError,match="exact transform"):
        rig.read(SimpleNamespace(frame=12,timestamp=SimpleNamespace(elapsed_seconds=.6)))


def test_none_destroy_return_succeeds_when_actor_is_actually_removed():
    actor = SimpleNamespace(id=41)
    world = SimpleNamespace(present=True,ticks=0)
    def destroy():
        world.present = False
        return None
    def tick():
        world.ticks += 1
    actor.destroy = destroy
    world.tick = tick
    world.get_actor = lambda identity:actor if world.present else None
    probe.destroy_probe_actor(actor,world)
    assert not world.present and world.ticks == 1
    probe.destroy_probe_actor(actor,world)  # No repeated destroy of a dead actor.
    assert world.ticks == 1


def test_false_destroy_return_does_not_hide_actor_still_present():
    actor = SimpleNamespace(id=41,destroy=lambda:False)
    world = SimpleNamespace(get_actor=lambda identity:actor,tick=lambda:None)
    with pytest.raises(RuntimeError,match="actor_destroy_failed"):
        probe.destroy_probe_actor(actor,world)


def test_obstacle_placement_uses_one_actor_and_never_destroys_during_setup():
    carla = pytest.importorskip("carla")
    from lane_dataset.geometry import LaneReference
    from lane_behavior.scenarios import obstacle_case
    x = np.arange(0.,241.)
    reference = LaneReference(np.c_[x,x*0,x*0],np.c_[x,np.full_like(x,4.5),x*0])
    actor = SimpleNamespace(id=10,bounding_box=carla.BoundingBox(carla.Location(),carla.Vector3D(2.,.8,.75)))
    operations=[]
    actor.set_simulate_physics=lambda value:operations.append(("physics",value))
    actor.get_transform=lambda:actor.transform
    def set_transform(value):
        operations.append(("placement",value))
        actor.pending_transform=value
    actor.set_transform=set_transform
    actor.destroy=lambda:pytest.fail("setup must not destroy its measured actor")
    bp=SimpleNamespace(id="vehicle.audi.a2",has_attribute=lambda name:False)
    def spawn(blueprint,transform):
        operations.append(("spawn",blueprint.id))
        actor.transform=carla.Transform()  # Stale just-spawned client snapshot.
        actor.pending_transform=transform
        return actor
    def tick():
        operations.append(("tick",None))
        actor.transform=actor.pending_transform
    world=SimpleNamespace(get_blueprint_library=lambda:SimpleNamespace(filter=lambda value:[bp]),spawn_actor=spawn,
        tick=tick,cast_ray=lambda *args:[SimpleNamespace(label=carla.CityObjectLabel.Roads,location=carla.Location(z=.05))])
    actors=[]
    actual,spec=probe._spawn_obstacle(world,dict(unmarked_behavior=obstacle_case("parked_car",240.,4.5)),reference,actors)
    assert actual is actor and actors == [actor]
    assert [op[0] for op in operations] == ["spawn","physics","tick","placement","tick"]
    vertices=np.array([[v.x,v.y,v.z] for v in actor.bounding_box.get_world_vertices(actor.transform)])
    assert vertices[:,1].min() == pytest.approx(4.5-spec["maximum_road_intrusion_m"],abs=1e-6)
    assert vertices[:,2].min() == pytest.approx(.05,abs=1e-6)
