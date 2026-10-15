"""Conservative production rendering profiles and full packed-frame checks.

These checks detect degenerate/frozen camera payloads. They are not a general
visual-artifact classifier. Wet reflections remain outside production profiles
after a same-pose CARLA comparison reproduced block artifacts in source RGB.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

RENDER_POLICY_VERSION = 1
WEATHER_FIELDS = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wetness',
    'sun_altitude_angle', 'sun_azimuth_angle', 'fog_density', 'fog_distance',
    'fog_falloff', 'wind_intensity')


def apply_production_weather(world, carla, appearance):
    profile = 'dry_cloudy_noon' if int(appearance) % 2 else 'dry_clear_noon'
    preset = carla.WeatherParameters.CloudyNoon if int(appearance) % 2 else carla.WeatherParameters.ClearNoon
    # Construct a fresh value: CARLA presets are shared mutable Python objects.
    weather = carla.WeatherParameters()
    for name in WEATHER_FIELDS:
        setattr(weather, name, float(getattr(preset, name)))
    weather.precipitation = 0.
    weather.precipitation_deposits = 0.
    weather.wetness = 0.
    expected = {name: float(getattr(weather, name)) for name in WEATHER_FIELDS}
    world.set_weather(weather)
    readbacks = []
    for tick in range(6):
        # Weather updates can become visible on the next simulator frame.
        # Read a single consistent value and allow bounded settling before
        # sensors/recording; a persistent mismatch still rejects the capture.
        observed = world.get_weather()
        actual = {name: float(getattr(observed, name)) for name in WEATHER_FIELDS}
        readbacks.append(actual)
        mismatch = {name: dict(requested=value, actual=actual[name]) for name, value in expected.items()
            if not math.isclose(actual[name], value, rel_tol=1e-6, abs_tol=1e-6)}
        if not mismatch:
            break
        if tick == 5:
            raise RuntimeError('production weather readback mismatch after five frames: '+str(mismatch))
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            world.wait_for_tick(10.)
    return dict(schema_version=RENDER_POLICY_VERSION, profile=profile,
        requested=expected, actual=actual, settling_ticks=tick, readbacks=readbacks,
        limitation='dry daylight only; wet reflection rendering not qualified')


def render_profile_errors(manifest):
    profile = manifest.get('render_profile')
    if not isinstance(profile, dict):
        return ['render_profile_readback_missing']
    expected_name = 'dry_cloudy_noon' if int(manifest.get('appearance', 0)) % 2 else 'dry_clear_noon'
    if profile.get('schema_version') != RENDER_POLICY_VERSION or profile.get('profile') != expected_name:
        return ['render_profile_invalid']
    expected, actual = profile.get('requested'), profile.get('actual')
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return ['render_profile_readback_invalid']
    for name in WEATHER_FIELDS:
        try:
            a, b = float(expected[name]), float(actual[name])
            if not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
                return ['render_profile_readback_mismatch']
        except (KeyError, TypeError, ValueError):
            return ['render_profile_readback_invalid']
    if any(actual[name] != 0. for name in ('precipitation', 'precipitation_deposits', 'wetness')):
        return ['unqualified_wet_rendering_profile']
    if actual['sun_altitude_angle'] < 30.:
        return ['unqualified_low_sun_rendering_profile']
    return []


class PackedVisualCheck:
    def __init__(self):
        self.count = 0
        self.errors = []
        self.previous = None
        self.repeated_moving = 0
        self.maximum_repeated_moving = 0

    def observe(self, packed, moving):
        array = np.asarray(packed)
        index = self.count
        self.count += 1
        if array.dtype != np.uint8 or array.shape != (6, 128, 256):
            self.errors.append(dict(index=index, reason='invalid_packed_frame'))
            return
        # The first four packed planes contain all luma pixels, without resampling.
        hist = np.bincount(array[:4].ravel(), minlength=256)
        cumulative = np.cumsum(hist)
        pixels = int(cumulative[-1])
        lo, hi = np.searchsorted(cumulative, [pixels*.01, pixels*.99])
        if hi-lo < 4 or hist[:18].sum()/pixels > .98 or hist[233:].sum()/pixels > .98:
            self.errors.append(dict(index=index, reason='degenerate_camera_frame'))
        same = self.previous is not None and np.array_equal(array, self.previous)
        self.repeated_moving = self.repeated_moving+1 if moving and same else 0
        self.maximum_repeated_moving = max(self.maximum_repeated_moving, self.repeated_moving)
        if self.repeated_moving == 5:
            self.errors.append(dict(index=index, reason='frozen_camera_while_moving'))
        self.previous = array.copy()

    def report(self):
        return dict(frame_count=self.count, passed=not self.errors, errors=self.errors,
            maximum_repeated_moving_frames=self.maximum_repeated_moving)


def validate_visual_capture(shard, manifest, telemetry):
    from .preprocessing import reference_module
    Reader = reference_module('frame_archive_v2').FrameArchiveReader
    errors = render_profile_errors(manifest)
    cameras = {}
    speeds = np.asarray(telemetry['speed'])
    for camera in ('narrow', 'wide'):
        entry = manifest['archives'][camera]
        path = (Path(shard) / entry['path']).resolve()
        if Path(shard).resolve() not in path.parents:
            raise ValueError('visual archive escaped shard')
        reader = Reader(path, expected_archive_sha256=entry['sha256'])
        check = PackedVisualCheck()
        try:
            if len(reader) != len(speeds):
                raise ValueError('visual archive/telemetry length mismatch')
            for index in range(len(reader)):
                check.observe(reader[index], moving=bool(speeds[index] > .5))
        finally:
            reader.close()
        cameras[camera] = {**check.report(), 'archive_sha256': entry['sha256']}
        errors.extend(camera+':'+item['reason'] for item in check.errors)
    return dict(schema_version=1, passed=not errors, errors=sorted(set(errors)), cameras=cameras,
        scope='all lossless packed frames: degenerate images and frozen moving camera; verified dry weather profile',
        limitation='does not certify absence of arbitrary rendering artifacts')
