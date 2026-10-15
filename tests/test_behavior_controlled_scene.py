from types import SimpleNamespace

import pytest

from lane_behavior.capture import controlled_scene_record


def scene(extra=()):
    ego=SimpleNamespace(id=1,type_id='vehicle.tesla.model3')
    obstacle=SimpleNamespace(id=2,type_id='vehicle.audi.a2')
    sensor=SimpleNamespace(id=3,type_id='sensor.camera.rgb')
    actors=[ego,obstacle,sensor,*extra]
    world=SimpleNamespace(get_actors=lambda:actors)
    snapshot=SimpleNamespace(frame=100,timestamp=SimpleNamespace(elapsed_seconds=5.))
    return world,snapshot,ego,obstacle


def test_full_inventory_retains_nontraffic_and_exact_traffic_states(monkeypatch):
    monkeypatch.setattr('lane_behavior.capture.actor_document',lambda actor,snapshot:dict(actor_id=actor.id,frame_id=snapshot.frame))
    result=controlled_scene_record(*scene())
    assert [a['actor_id'] for a in result['actors']]==[1,2,3]
    assert [a['state'] is not None for a in result['actors']]==[True,True,False]
    assert result['static_obstacle_actor_ids']==[2]
    assert result['scope']=='simulator_scene_QA_not_visual_free_space'


@pytest.mark.parametrize('kind',['vehicle.audi.tt','walker.pedestrian.0001'])
def test_extra_vehicle_or_pedestrian_rejects_controlled_static_fixture(monkeypatch,kind):
    monkeypatch.setattr('lane_behavior.capture.actor_document',lambda actor,snapshot:dict(actor_id=actor.id))
    with pytest.raises(RuntimeError,match='unexpected traffic'):
        controlled_scene_record(*scene([SimpleNamespace(id=4,type_id=kind)]))


def test_missing_declared_obstacle_rejects_scene(monkeypatch):
    monkeypatch.setattr('lane_behavior.capture.actor_document',lambda actor,snapshot:dict(actor_id=actor.id))
    world,snapshot,ego,obstacle=scene()
    world.get_actors=lambda:[ego]
    with pytest.raises(RuntimeError,match='missing'):
        controlled_scene_record(world,snapshot,ego,obstacle)
