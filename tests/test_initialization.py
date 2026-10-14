import numpy as np
import pytest
from lane_dataset.controller import SteeringCalibration
from lane_dataset.geometry import LaneReference
from lane_dataset.initialization import PhysicalInitializer


def test_map_initializer_cannot_produce_retained_recording_controls():
    s=np.arange(0.,30.)
    ref=LaneReference(np.column_stack([s,-np.ones_like(s)*1.7,np.zeros_like(s)]),
                      np.column_stack([s,np.ones_like(s)*1.7,np.zeros_like(s)]),s)
    row=dict(position_world=[0.,.2,0.],rotation_world=[0.,0.,0.],speed=0.)
    driver=PhysicalInitializer(SteeringCalibration.fixture())
    command=driver.control(row,ref,1.,recording=False)
    assert command['steer']<0 and 0<command['throttle']<=.02
    with pytest.raises(ValueError,match='prohibited'):
        driver.control(row,ref,1.,recording=True)
