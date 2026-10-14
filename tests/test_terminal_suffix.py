import pytest
from lane_dataset.terminal_suffix import validate_terminal_stop,suffix_stop_control


def test_stop_requires_complete_maneuver_and_five_meter_reserve():
    phases={name:[i*10.,(i+1)*10.] for i,name in enumerate(('approach','entry','apex','exit','recovery'))}
    assert validate_terminal_stop({'maneuver_phase_ranges':phases,'terminal_stop_route_m':55.})==55.
    for stop in (49.,54.99,float('nan')):
        with pytest.raises(ValueError):validate_terminal_stop({'maneuver_phase_ranges':phases,'terminal_stop_route_m':stop})
    assert validate_terminal_stop(None) is None


def test_suffix_stop_preserves_actual_steering_and_bounds_brake_increment():
    row=dict(throttle=.3,steer=-.08,brake=.12)
    control=suffix_stop_control(row)
    assert control==dict(throttle=0.,steer=-.08,brake=pytest.approx(.17))
    assert suffix_stop_control(dict(steer=.02,brake=.49))['brake']==.5
