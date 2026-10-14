"""Measured rolling initialization gate, separate from recorded corpus frames."""
from __future__ import annotations

from dataclasses import dataclass
import math


def check_recording_seam(last_bootstrap,first_recorded,dt_s=.05,
                         maximum_lateral_acceleration_mps2=3.5,maximum_lateral_jerk_mps3=5.):
    """Score the unfiltered physical derivative across the archive boundary.

    Both inputs are complete ``simulator.state`` rows. A missing frame cannot
    reduce the measured derivative or qualify the transition into recording.
    """
    values=[]
    for row in (last_bootstrap,first_recorded):
        yaw=float(row["rotation_world"][2]);acc=row["acceleration_world"]
        if len(acc)!=3 or not all(math.isfinite(float(value)) for value in (*acc,yaw,row["timestamp"])):
            raise ValueError("recording seam needs finite measured physical telemetry")
        values.append(-math.sin(yaw)*float(acc[0])+math.cos(yaw)*float(acc[1]))
    interval=float(first_recorded["timestamp"])-float(last_bootstrap["timestamp"])
    errors=[]
    if first_recorded["frame_id"]!=last_bootstrap["frame_id"]+1 or not math.isclose(interval,dt_s,rel_tol=0,abs_tol=1e-6):
        errors.append("bootstrap_recording_seam_missing_duplicate_or_non20hz_frame")
    jerk=(values[1]-values[0])/interval if interval>0 else None
    if max(abs(value) for value in values)>maximum_lateral_acceleration_mps2:
        errors.append("bootstrap_recording_seam_lateral_acceleration_above_limit")
    if jerk is None or abs(jerk)>maximum_lateral_jerk_mps3:
        errors.append("bootstrap_recording_seam_lateral_jerk_above_limit")
    return {"passed":not errors,"errors":errors,"last_bootstrap_frame_id":int(last_bootstrap["frame_id"]),
        "first_recorded_frame_id":int(first_recorded["frame_id"]),"interval_s":interval,
        "lateral_acceleration_mps2":values,"lateral_jerk_mps3":jerk,
        "scope":"raw measured derivative across initialization-to-dense-recording boundary"}


@dataclass
class RollingWarmup:
    speed_min_mps: float
    speed_max_mps: float
    dt_s: float = .05
    steady_duration_s: float = .5
    maximum_frames: int = 400
    maximum_lateral_acceleration_mps2: float = 3.5
    maximum_lateral_jerk_mps3: float = 5.

    def __post_init__(self):
        if not 0<self.speed_min_mps<=self.speed_max_mps:
            raise ValueError("rolling warmup needs a positive measured speed domain")
        if not all(math.isfinite(value) and value>0 for value in (self.dt_s,self.steady_duration_s,
            self.maximum_lateral_acceleration_mps2,self.maximum_lateral_jerk_mps3)) or self.maximum_frames<2:
            raise ValueError("rolling warmup needs finite positive timing and physical limits")
        self.frame_count=0;self.consecutive_intervals=0;self.ready=False
        self.previous_timestamp=None;self.previous_lateral=None
        self.peak_lateral=0.;self.peak_jerk=0.;self.last_jerk=None

    def observe(self,timestamp_s,speed_mps,lateral_acceleration_mps2,*,command_accepted,visible_reference):
        if self.ready:raise RuntimeError("rolling warmup already completed")
        if not all(math.isfinite(value) for value in (timestamp_s,speed_mps,lateral_acceleration_mps2)):
            raise ValueError("warmup physical telemetry must be finite")
        self.frame_count+=1;jerk=None
        if self.previous_timestamp is not None:
            interval=timestamp_s-self.previous_timestamp
            if not math.isclose(interval,self.dt_s,rel_tol=0,abs_tol=1e-6):
                raise ValueError("warmup missing/duplicate/non20Hz physical frame")
            jerk=(lateral_acceleration_mps2-self.previous_lateral)/interval
            self.peak_jerk=max(self.peak_jerk,abs(jerk))
        self.peak_lateral=max(self.peak_lateral,abs(lateral_acceleration_mps2));self.last_jerk=jerk
        valid=(jerk is not None and abs(jerk)<=self.maximum_lateral_jerk_mps3 and
               abs(lateral_acceleration_mps2)<=self.maximum_lateral_acceleration_mps2 and
               self.speed_min_mps<=speed_mps<=self.speed_max_mps and command_accepted and visible_reference)
        self.consecutive_intervals=self.consecutive_intervals+1 if valid else 0
        self.ready=self.consecutive_intervals*self.dt_s>=self.steady_duration_s-1e-9
        self.previous_timestamp=timestamp_s;self.previous_lateral=lateral_acceleration_mps2
        if not self.ready and self.frame_count>=self.maximum_frames:
            raise TimeoutError(f"rolling warmup failed measured speed/jerk/visibility gate within {self.maximum_frames} frames")
        return self.ready

    def report(self):
        return {"ready":self.ready,"frames":self.frame_count,
                "steady_duration_s":self.consecutive_intervals*self.dt_s,
                "required_steady_duration_s":self.steady_duration_s,
                "measured_speed_domain_mps":[self.speed_min_mps,self.speed_max_mps],
                "maximum_observed_lateral_acceleration_mps2":self.peak_lateral,
                "maximum_observed_lateral_jerk_mps3":self.peak_jerk,
                "maximum_frames":self.maximum_frames,
                "scope":"physics-only initialization; retained separately, never counted as corpus frames or labels"}
