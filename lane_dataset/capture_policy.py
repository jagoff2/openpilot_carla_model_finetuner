"""Versioned collection tolerances, separate from measured calibration data."""
from __future__ import annotations

import math

CAPTURE_POLICY_VERSION = 6
SPEED_TOLERANCE_MPS = .1


def speed_bounds(calibration, tolerance=SPEED_TOLERANCE_MPS):
    if not math.isfinite(tolerance) or not 0 <= tolerance <= SPEED_TOLERANCE_MPS:
        raise ValueError('invalid collection speed tolerance')
    return max(.3, calibration.speed_min_mps-tolerance), calibration.speed_max_mps+tolerance


def stopping_distance(speed, deceleration, delay):
    """Distance during reaction delay followed by constant physical braking."""
    if speed < 0 or deceleration <= 0 or delay < 0:
        raise ValueError('invalid stopping distance inputs')
    return speed*delay + speed*speed/(2*deceleration)
