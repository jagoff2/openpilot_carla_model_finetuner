"""Explicit unlabelled physical stop after a completed maneuver.

The dense future suffix can include a measured stop. It never needs to drive
into an unobservable next intersection merely to accumulate ten seconds.
These controls are not expert actions and causal plan masks stop before them.
"""
import math


def validate_terminal_stop(task):
    stop = None if task is None else task.get('terminal_stop_route_m')
    if stop is None:return None
    phases=task.get('maneuver_phase_ranges',{})
    if (not isinstance(stop,(int,float)) or not math.isfinite(stop)
            or set(phases)!={'approach','entry','apex','exit','recovery'}
            or stop<float(phases['recovery'][1])+5.):
        raise ValueError('terminal suffix stop must follow the complete maneuver by at least five meters')
    return float(stop)


def suffix_stop_control(row):
    # Preserve actual steering and ramp braking. No geometry is consulted.
    return dict(throttle=0.,brake=min(.5,float(row['brake'])+.05),steer=float(row['steer']))
