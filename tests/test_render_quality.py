from types import SimpleNamespace

import numpy as np
import pytest

from lane_dataset.render_quality import (WEATHER_FIELDS, apply_production_weather,
    render_profile_errors, PackedVisualCheck)


def test_weather_is_dry_read_back_and_does_not_mutate_presets():
    class Weather:
        def __init__(self):
            for name in WEATHER_FIELDS:
                setattr(self, name, 0.)
    Weather.ClearNoon = Weather()
    Weather.ClearNoon.sun_altitude_angle = 75.
    Weather.CloudyNoon = Weather()
    Weather.CloudyNoon.sun_altitude_angle = 45.
    Weather.CloudyNoon.wetness = 70.
    world = SimpleNamespace()
    world.set_weather = lambda weather: setattr(world, 'weather', weather)
    world.get_weather = lambda: world.weather
    profile = apply_production_weather(world, SimpleNamespace(WeatherParameters=Weather), 1)
    assert profile['actual']['wetness'] == 0.
    assert Weather.CloudyNoon.wetness == 70.
    assert not render_profile_errors(dict(appearance=1, render_profile=profile))
    profile['actual']['wetness'] = profile['requested']['wetness'] = 70.
    assert render_profile_errors(dict(appearance=1, render_profile=profile)) == ['unqualified_wet_rendering_profile']
    assert render_profile_errors(dict(appearance=1)) == ['render_profile_readback_missing']


def test_frozen_moving_camera_and_black_frame_fail_without_rejecting_stationary_tail():
    rng = np.random.default_rng(1)
    frame = rng.integers(16, 235, (6,128,256), dtype=np.uint8)
    moving, stopped = PackedVisualCheck(), PackedVisualCheck()
    for _ in range(6):
        moving.observe(frame, moving=True)
        stopped.observe(frame, moving=False)
    assert moving.report()['errors'] == [dict(index=5, reason='frozen_camera_while_moving')]
    assert stopped.report()['passed']
    stopped.observe(np.zeros_like(frame), moving=False)
    assert stopped.report()['errors'][-1]['reason'] == 'degenerate_camera_frame'


def test_different_valid_frames():
    check = PackedVisualCheck()
    rng = np.random.default_rng(2)
    for _ in range(10):
        check.observe(rng.integers(16,235,(6,128,256),dtype=np.uint8), moving=True)
    assert check.report()['passed']
    assert check.report()['frame_count'] == 10


@pytest.mark.parametrize('update_after',[1,3,None])
def test_delayed_weather_update_is_bounded_and_persistent_mismatch_rejects(update_after):
    class Weather:
        def __init__(self):
            for name in WEATHER_FIELDS:setattr(self,name,0.)
    Weather.ClearNoon=Weather();Weather.ClearNoon.sun_altitude_angle=45.
    Weather.CloudyNoon=Weather.ClearNoon
    class World:
        def __init__(self):self.current=Weather();self.ticks=0;self.reads=0
        def set_weather(self,value):self.pending=value
        def get_weather(self):self.reads+=1;return self.current
        def get_settings(self):return SimpleNamespace(synchronous_mode=True)
        def tick(self):
            self.ticks+=1
            if self.ticks==update_after:self.current=self.pending
    world=World();carla=SimpleNamespace(WeatherParameters=Weather)
    if update_after is None:
        with pytest.raises(RuntimeError,match='after five frames.*sun_altitude_angle'):
            apply_production_weather(world,carla,0)
        assert world.ticks==5 and world.reads==6
    else:
        report=apply_production_weather(world,carla,0)
        assert world.ticks==report['settling_ticks']==update_after
        assert len(report['readbacks'])==world.reads==update_after+1
        assert not render_profile_errors(dict(appearance=0,render_profile=report))


def test_whole_archive_check_binds_hashes_and_finds_a_frozen_stream(tmp_path):
    from lane_dataset.preprocessing import reference_module
    from lane_dataset.render_quality import validate_visual_capture, WEATHER_FIELDS
    from lane_dataset.config import sha256_file
    Writer=reference_module('frame_archive_v2').FrameArchiveWriter
    rng=np.random.default_rng(3)
    first=rng.integers(16,235,(6,128,256),dtype=np.uint8)
    entries={}
    for camera in ('narrow','wide'):
        path=tmp_path/(camera+'.npz')
        writer=Writer(path,dtype=np.uint8,frame_shape=first.shape,expected_count=6)
        for index in range(6):
            writer.write(index,first if camera=='wide' else rng.integers(16,235,first.shape,dtype=np.uint8))
        writer.finalize(complete=True)
        entries[camera]=dict(path=path.name,sha256=sha256_file(path))
    weather={name:0. for name in WEATHER_FIELDS};weather['sun_altitude_angle']=45.
    manifest=dict(appearance=1,archives=entries,render_profile=dict(schema_version=1,
        profile='dry_cloudy_noon',requested=weather,actual=weather.copy()))
    report=validate_visual_capture(tmp_path,manifest,dict(speed=np.ones(6)))
    assert report['errors']==['wide:frozen_camera_while_moving']
    assert report['cameras']['narrow']['passed']
    assert report['cameras']['wide']['archive_sha256']==entries['wide']['sha256']
