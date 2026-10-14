import numpy as np
import pytest

from lane_dataset.calibrate_extended import pulse_schedule


def test_expanded_calibration_excites_both_turn_directions_and_balances_heading():
    pulse=pulse_schedule(amplitudes=(.05,.1,.2,.25,.3,.35,.4))
    assert len(pulse)==1890
    assert np.max(pulse)==pytest.approx(.4)
    assert np.min(pulse)==pytest.approx(-.4)
    assert np.sum(pulse)==pytest.approx(0.,abs=1e-12)
    assert len(pulse_schedule())==1080


@pytest.mark.parametrize("amplitudes",[(),(0.,),(-.1,),(.1,float("nan")),(1.01,)])
def test_invalid_steering_domain_is_rejected(amplitudes):
    with pytest.raises(ValueError,match="steering fractions"):
        pulse_schedule(amplitudes=amplitudes)
