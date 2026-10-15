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


def test_initializer_keeps_progress_on_first_lap_of_overlapping_route():
    angle=np.linspace(0.,4*np.pi,801);radius=20.
    center=np.c_[radius*np.cos(angle),radius*np.sin(angle),angle*0]
    radial=np.c_[np.cos(angle),np.sin(angle),angle*0]
    ref=LaneReference(center+1.75*radial,center-1.75*radial,radius*angle)
    calibration=SteeringCalibration.fixture();driver=PhysicalInitializer(calibration)
    values=[]
    for a in np.linspace(0.,1.,40):
        yaw=a+np.pi/2;rear=radius*np.array([np.cos(a),np.sin(a)])
        actor=rear+calibration.reference_offset_m*np.array([np.cos(yaw),np.sin(yaw)])
        driver.control(dict(position_world=[*actor,0.],rotation_world=[0.,0.,yaw],speed=3.5),ref,3.5,recording=False)
        values.append(driver.progress_s)
    assert values[0]<.001 and values[-1]==pytest.approx(20.,abs=.001)
    assert np.max(np.diff(values))<1.
