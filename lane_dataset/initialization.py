"""Explicit map-based physical initialization, prohibited in retained recording."""
import math
import numpy as np


class PhysicalInitializer:
    def __init__(self,calibration):
        self.calibration=calibration;self.throttle=0.;self.steer=0.;self.progress_s=None

    def control(self,row,reference,target_speed,*,recording):
        if recording:raise ValueError('map initialization is prohibited in retained recording')
        self.steer=float(row.get('steer',self.steer));self.throttle=float(row.get('throttle',self.throttle))
        c=self.calibration;yaw=float(row['rotation_world'][2])
        heading=np.array([math.cos(yaw),math.sin(yaw)])
        rear=np.asarray(row['position_world'][:2])-c.reference_offset_m*heading
        progress,_=reference.project(rear,reference.s[0] if self.progress_s is None else self.progress_s)
        self.progress_s=progress
        target=reference._center(min(reference.s[-1],progress+5.))[:2]
        delta=target-rear
        lateral=-math.sin(yaw)*delta[0]+math.cos(yaw)*delta[1]
        curvature=2*lateral/max(float(delta@delta),1.)
        angle=math.atan(c.wheelbase_m*curvature)
        angle=float(np.clip(angle,c.angle_from_steer(c.steering_fraction_min),c.angle_from_steer(c.steering_fraction_max)))
        desired=c.steer_from_angle(angle)
        self.steer=float(np.clip(desired,self.steer-.02,self.steer+.02))
        demand=.20*(target_speed-row['speed'])+.1
        brake=float(np.clip(-demand,0,.5))
        self.throttle=0. if brake else float(np.clip(demand,max(0,self.throttle-.02),min(.4,self.throttle+.02)))
        return dict(throttle=self.throttle,brake=brake,steer=self.steer)
