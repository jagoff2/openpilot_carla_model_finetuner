from types import SimpleNamespace
import pytest
from lane_dataset.simulator import ExactSensorQueue, world_owner, verify_camera_attributes


def test_generated_world_physics_are_applied_and_read_back():
    from lane_dataset.simulator import apply_checked_settings
    values=SimpleNamespace(synchronous_mode=False,fixed_delta_seconds=None,no_rendering_mode=True)
    applied=[]
    world=SimpleNamespace(get_settings=lambda:values,apply_settings=lambda item:applied.append(vars(item).copy()))
    expected=dict(synchronous_mode=True,fixed_delta_seconds=.05,no_rendering_mode=False)
    assert apply_checked_settings(world,expected)==expected
    assert applied==[expected]
    world.apply_settings=lambda item:setattr(values,"fixed_delta_seconds",None)
    with pytest.raises(RuntimeError,match="readback mismatch"):
        apply_checked_settings(world,expected)


def test_exact_frame_and_timestamp_rejection():
    q = ExactSensorQueue("test")
    q.callback(SimpleNamespace(frame=3,timestamp=.15))
    q.get(3,.15)
    q.callback(SimpleNamespace(frame=5,timestamp=.25))
    with pytest.raises(RuntimeError,match="non-consecutive"):
        q.get(5,.25)
    q = ExactSensorQueue("test")
    q.callback(SimpleNamespace(frame=1,timestamp=.1))
    with pytest.raises(RuntimeError,match="timestamp"):
        q.get(1,.05)


def test_world_lock_excludes_other_tick_owner(tmp_path):
    with world_owner(2500,tmp_path):
        with pytest.raises(OSError):
            with world_owner(2500,tmp_path):
                pass


def test_camera_readback_rejects_projection_or_photometric_drift():
    requested=dict(fov="79.812347123",lens_k="0.0",enable_postprocess_effects="true")
    actual=dict(fov="79.812347",lens_k="0",enable_postprocess_effects="True")
    verify_camera_attributes(requested,actual)
    for changed in ({"lens_k":"-1"},{"enable_postprocess_effects":"false"},
                    {"fov":"90"},{"fov":"nan"}):
        with pytest.raises(RuntimeError,match="readback mismatch"):
            verify_camera_attributes(requested,{**actual,**changed})
    with pytest.raises(RuntimeError,match="readback mismatch"):
        verify_camera_attributes(requested,{})
