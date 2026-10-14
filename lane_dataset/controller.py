"""Constrained, delay-aware bicycle MPC and empirical steering identification.

No simulator transforms, velocity overrides, or full-map references are used.
Fixture calibrations are explicitly unqualified. Physical quality qualification
is a separate gate and cannot be inferred from optimizer success.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import least_squares, minimize

from .geometry import ObservableReference, wrap_angle


@dataclass(frozen=True)
class VehicleState:
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float
    yaw_rate_rps: float
    steering_rad: float | None = None
    timestamp_s: float = 0.
    lateral_acceleration_mps2: float | None = None


@dataclass(frozen=True)
class SteeringCalibration:
    wheelbase_m: float
    steering_gain_rad: float
    max_steering_rad: float
    delay_s: float
    time_constant_s: float
    speed_min_mps: float
    speed_max_mps: float
    validation_yaw_rate_rmse_rps: float
    qualified: bool
    sample_count: int
    source: str = "CARLA measured steering maneuvers"
    validation_peak_yaw_rate_error_rps: float = float("inf")
    steering_fraction_min: float = -1.
    steering_fraction_max: float = 1.
    steering_cubic_rad: float = 0.
    response_rate_rps: float = 100.
    reference_offset_m: float = 0.

    @classmethod
    def fixture(cls, wheelbase_m=2.8, steering_gain_rad=.65, delay_s=.1, time_constant_s=.15):
        return cls(wheelbase_m, steering_gain_rad, abs(steering_gain_rad), delay_s,
                   time_constant_s, 0., 30., 0., False, 0, "analytical fixture, not CARLA qualified", 0.)

    def to_dict(self):
        return asdict(self)

    def angle_from_steer(self, fraction):
        return self.steering_gain_rad*fraction+self.steering_cubic_rad*fraction**3

    def steer_from_angle(self, angle):
        if abs(self.steering_cubic_rad)<1e-10:return float(angle/self.steering_gain_rad)
        roots=np.roots([self.steering_cubic_rad,0.,self.steering_gain_rad,-angle])
        legal=[float(root.real) for root in roots if abs(root.imag)<1e-8 and
               self.steering_fraction_min-1e-6<=root.real<=self.steering_fraction_max+1e-6]
        if not legal:raise ValueError("requested angle outside measured monotone steering response")
        return min(legal,key=abs)


def _advance_steering(motor, effective, target, dt, tau, rate):
    """Exact first-order response to a slew-limited actuator ramp over one step."""
    if rate>=99.:
        return target, effective+(1-math.exp(-dt/tau))*(target-effective)
    signed_rate=math.copysign(rate,target-motor)
    ramp_time=min(dt,abs(target-motor)/rate)
    final_motor=motor+signed_rate*ramp_time
    at_ramp=(motor+signed_rate*(ramp_time-tau)+
             (effective-motor+signed_rate*tau)*math.exp(-ramp_time/tau))
    return final_motor,final_motor+(at_ramp-final_motor)*math.exp(-(dt-ramp_time)/tau)


def reference_lateral_acceleration(speeds_mps,yaw_rates_rps,dt_s,reference_offset_m,initial_acceleration_mps2):
    """Actor-frame lateral acceleration, including the offset yaw transient.

    The calibrated actor lateral velocity is offset*yaw_rate, so its inertial
    lateral acceleration is d(offset*yaw_rate)/dt + body_forward_speed*yaw_rate.
    Using only speed*yaw_rate misses abrupt steering transients at this point.
    """
    speed=np.asarray(speeds_mps,float);rate=np.asarray(yaw_rates_rps,float)
    if speed.shape!=rate.shape or speed.ndim!=1 or len(speed)<2 or dt_s<=0:
        raise ValueError("lateral acceleration requires matching timed state sequences")
    lateral_velocity=reference_offset_m*rate
    forward_speed=np.sqrt(np.maximum(0.,speed**2-lateral_velocity**2))
    result=np.empty_like(speed);result[0]=initial_acceleration_mps2
    result[1:]=forward_speed[1:]*rate[1:]+np.diff(lateral_velocity)/dt_s
    return result


def fit_rate_limited_calibration(timestamps_s, steering_fractions, speeds_mps, yaw_rates_rps,
                                wheelbase_m, max_steering_rad,max_delay_s=.4,validation_fraction=.3):
    """Identify the observed slew-rate + lag + cubic static response, on training only.

    Held-out gates remain identical to the linear calibration. This fit captures
    amplitude-dependent effective bicycle response; it is restricted to the
    measured speed and command range. It does not extrapolate tire dynamics.
    """
    t,u,v,r=[np.asarray(value,float) for value in (timestamps_s,steering_fractions,speeds_mps,yaw_rates_rps)]
    # Reuse all timing/shape/physical-input validation from the simple fit.
    baseline=fit_steering_calibration(t,u,v,r,wheelbase_m,max_steering_rad,max_delay_s,validation_fraction)
    dt=float(np.median(np.diff(t)));cut=int(len(t)*(1-validation_fraction));delta=np.arctan(wheelbase_m*r/v)
    candidates=[]
    for delay_ticks in range(int(round(max_delay_s/dt))+1):
        delayed_u=u[np.maximum(np.arange(len(t))-delay_ticks,0)]
        def predict(parameters):
            gain,tau,rate,cubic=parameters;output=np.empty(len(t));output[0]=delta[0];motor=delta[0]
            for i in range(1,len(t)):
                target=gain*delayed_u[i-1]+cubic*delayed_u[i-1]**3
                motor,output[i]=_advance_steering(motor,output[i-1],target,dt,tau,rate)
            return output
        solved=least_squares(lambda p:predict(p)[1:cut]-delta[1:cut],
            [baseline.steering_gain_rad,.03,2.8,-.5],
            bounds=([.02,.003,.2,-8.],[max_steering_rad*1.5,1.5,20.,8.]),loss="soft_l1",f_scale=.01)
        predicted=predict(solved.x)
        train_error=float(np.mean((predicted[1:cut]-delta[1:cut])**2))
        candidates.append((train_error,delay_ticks,solved.x,predicted))
    _,ticks,(gain,tau,rate,cubic),predicted=min(candidates,key=lambda entry:entry[0])
    residual=v*np.tan(predicted)/wheelbase_m-r
    rmse=float(np.sqrt(np.mean(residual[cut:]**2)));peak=float(np.max(np.abs(residual[cut:])))
    signal_rms=float(np.sqrt(np.mean(r[cut:]**2)))
    excitation=all(np.any(u[part]>.01) and np.any(u[part]<-.01) for part in (slice(0,cut),slice(cut,None)))
    monotone=np.min(gain+3*cubic*np.linspace(u.min(),u.max(),101)**2)>.01
    qualified=bool(excitation and np.sum(u*r)>0 and rmse<=min(.025,.25*signal_rms) and peak<=.075
        and ticks*dt<max_delay_s and .021<gain<=max_steering_rad*1.2 and .00301<tau<1.49
        and .201<rate<19.99 and abs(cubic)<7.99 and monotone)
    return SteeringCalibration(float(wheelbase_m),float(gain),float(max_steering_rad),float(ticks*dt),
        float(tau),float(v.min()),float(v.max()),rmse,qualified,len(t),
        source="CARLA measured steering; slew-limited cubic response plus first-order lag",
        validation_peak_yaw_rate_error_rps=peak,steering_fraction_min=float(u.min()),steering_fraction_max=float(u.max()),
        steering_cubic_rad=float(cubic),response_rate_rps=float(rate))


def fit_steering_calibration(timestamps_s, steering_fractions, speeds_mps, yaw_rates_rps,
                            wheelbase_m: float, max_steering_rad: float,
                            max_delay_s=.4, validation_fraction=.3):
    """Identify gain, pure delay and first-order steering response from telemetry.

    CARLA inputs and measured yaw use positive-right conventions. The final
    chronological segment is held out from fitting. Require both steering signs
    in both segments and modest speeds; root must additionally run road-quality
    qualification before accepting generated episodes. Samples reflect the
    command held on [t[i], t[i+1]) and measured state at t[i].
    """
    t, u, v, r = [np.asarray(item, float) for item in
                  (timestamps_s, steering_fractions, speeds_mps, yaw_rates_rps)]
    if len(t) < 80 or any(a.shape != t.shape for a in (u, v, r)):
        raise ValueError("calibration needs at least 80 matching samples")
    if not np.isfinite(np.stack((t,u,v,r))).all() or np.any(np.diff(t) <= 0) or np.any(v < 1.):
        raise ValueError("calibration needs finite chronological forward-motion samples >=1 m/s")
    dt = float(np.median(np.diff(t)))
    if np.max(np.abs(np.diff(t) - dt)) > dt*.02:
        raise ValueError("calibration frame timing is not uniform")
    if wheelbase_m <= 0 or max_steering_rad <= 0 or max_delay_s < 0:
        raise ValueError("invalid physical steering limits")
    cut = int(len(t)*(1-validation_fraction))
    if min(cut, len(t)-cut) < 20:
        raise ValueError("insufficient calibration/held-out segment")
    measured_delta = np.arctan(wheelbase_m*r/v)
    candidates = []
    for delay_ticks in range(int(round(max_delay_s/dt))+1):
        delayed_u = u[np.maximum(np.arange(len(t))-delay_ticks, 0)]
        def predict(parameters):
            gain, tau = parameters
            result = np.empty(len(t)); result[0] = measured_delta[0]
            alpha = 1-math.exp(-dt/tau)
            for i in range(1, len(t)):
                result[i] = result[i-1] + alpha*(gain*delayed_u[i-1]-result[i-1])
            return result
        fitted = least_squares(lambda p: predict(p)[1:cut]-measured_delta[1:cut],
            [max_steering_rad*.8, .15], bounds=([.02, dt*.25], [max_steering_rad*1.5, 1.5]),
            loss="soft_l1", f_scale=.01)
        train_error = float(np.mean((predict(fitted.x)[1:cut]-measured_delta[1:cut])**2))
        candidates.append((train_error, delay_ticks, fitted.x, predict(fitted.x)))
    _, ticks, (gain,tau), delta_hat = min(candidates, key=lambda item:item[0])
    residual = v*np.tan(delta_hat)/wheelbase_m-r
    rmse = float(np.sqrt(np.mean(residual[cut:]**2)))
    peak = float(np.max(np.abs(residual[cut:])))
    excitation = all(np.any(u[part] > .01) and np.any(u[part] < -.01)
                     for part in (slice(0,cut), slice(cut,None)))
    sign_agrees = float(np.sum(u * r)) > 0
    signal_rms = float(np.sqrt(np.mean(r[cut:]**2)))
    qualified = bool(excitation and sign_agrees and rmse <= min(.025, .25*signal_rms) and peak <= .075
                     and ticks*dt < max_delay_s and .021 < gain <= max_steering_rad*1.2
                     and tau > dt*.251 and tau < 1.49)
    return SteeringCalibration(float(wheelbase_m), float(gain), float(max_steering_rad),
        float(ticks*dt), float(tau), float(v.min()), float(v.max()), rmse,
        qualified, len(t), validation_peak_yaw_rate_error_rps=peak,
        steering_fraction_min=float(u.min()), steering_fraction_max=float(u.max()))


@dataclass(frozen=True)
class MPCConfig:
    control_dt_s: float = .05
    prediction_dt_s: float = .15
    horizon_steps: int = 16
    max_lateral_acceleration_mps2: float = 2.5
    # Soft target for the fitted prediction model. Executed command knots have
    # a separate hard bound; physical recordings must still pass 5 m/s^3 QA.
    max_lateral_jerk_mps3: float = 3.6
    max_executed_command_jerk_mps3: float = 2.0
    max_steering_rate_rps: float = .65
    max_acceleration_mps2: float = 1.5
    min_acceleration_mps2: float = -3.
    vehicle_width_m: float = 1.85
    vehicle_length_m: float = 4.6
    clearance_m: float = .15
    max_iterations: int = 70
    require_calibration: bool = True
    speed_calibration_margin_mps: float = .1


@dataclass(frozen=True)
class MPCCommand:
    accepted: bool
    reasons: tuple[str, ...]
    curvature_1pm: float
    carla_steer: float
    steering_angle_rad: float
    acceleration_mps2: float
    target_speed_mps: float
    action_horizon_s: float
    predicted_states: np.ndarray
    evidence_frames: tuple
    optimizer_success: bool
    optimizer_message: str
    min_constraint_margin: float
    constraint_margins: dict | None = None


class BicycleMPC:
    def __init__(self, calibration: SteeringCalibration, config: MPCConfig | None = None):
        self.calibration = calibration
        self.config = config or MPCConfig()
        if self.config.require_calibration and not calibration.qualified:
            raise ValueError("production MPC requires measured held-out steering calibration")
        self._last_command_rad = 0.
        self._previous_command_rad = 0.
        self._history: list[tuple[float,float]] = []
        self._previous_solution = None
        self._last_timestamp = None
        self._last_observed_delta = None
        self._last_yaw_rate = None

    def reset(self):
        self._last_command_rad = 0.; self._history.clear()
        self._previous_command_rad = 0.
        self._previous_solution = None; self._last_timestamp = None
        self._last_observed_delta = None
        self._last_yaw_rate = None

    def _initial_steering(self, state):
        return float(state.steering_rad if state.steering_rad is not None else
                     np.arctan(self.calibration.wheelbase_m*state.yaw_rate_rps/max(state.speed_mps,.1)))

    def _rollout(self, state, variables, dense=False):
        c, p = self.config, self.calibration
        n = len(variables)//2; commands = variables[:n]; accelerations = variables[n:]
        h = c.prediction_dt_s
        command_times=np.arange(n)*h
        subdivisions = max(1, int(round(h/c.control_dt_s))); dt = h/subdivisions
        x,y,yaw,v = state.x_m,state.y_m,state.yaw_rad,state.speed_mps
        delta = self._initial_steering(state)
        motor=delta
        if self._last_observed_delta is not None and self._last_timestamp is not None:
            derivative=(delta-self._last_observed_delta)/max(state.timestamp_s-self._last_timestamp,1e-6)
            motor=float(np.clip(delta+p.time_constant_s*derivative,-p.max_steering_rad,p.max_steering_rad))
        rows = [[x,y,yaw,v,state.yaw_rate_rps,delta,0.]]
        for k in range(n):
            for j in range(subdivisions):
                time = k*h+j*dt
                source = time-p.delay_s
                if source < -1e-9:
                    old = [value for stamp,value in self._history if stamp <= state.timestamp_s+source+1e-9]
                    requested = old[-1] if old else self._initial_steering(state)
                else:
                    # Future control knots form a steering ramp. Replanning at
                    # 20 Hz executes its first command; zero-order 150 ms holds
                    # create artificial yaw-acceleration steps at a fast actor.
                    requested=float(np.interp(max(0,source),command_times,commands))
                motor,delta=_advance_steering(motor,delta,requested,dt,p.time_constant_s,p.response_rate_rps)
                next_v = max(0.,v+accelerations[k]*dt)
                mid_v = .5*(v+next_v)
                yaw_rate = mid_v*math.tan(delta)/p.wheelbase_m
                beta=math.asin(max(-.95,min(.95,p.reference_offset_m*math.tan(delta)/p.wheelbase_m)))
                x += mid_v*math.cos(yaw+beta+.5*yaw_rate*dt)*dt
                y += mid_v*math.sin(yaw+beta+.5*yaw_rate*dt)*dt
                yaw += yaw_rate*dt; v=next_v
                if dense:
                    rows.append([x,y,yaw,v,v*math.tan(delta)/p.wheelbase_m,delta,time+dt])
            if not dense:
                rows.append([x,y,yaw,v,v*math.tan(delta)/p.wheelbase_m,delta,(k+1)*h])
        return np.asarray(rows,float)

    def control(self, state: VehicleState, reference: ObservableReference, desired_speed_mps: float, *, desired_offset_m=0.):
        if not isinstance(reference, ObservableReference):
            raise TypeError("MPC accepts ObservableReference only; full map geometry is prohibited")
        if not math.isfinite(desired_offset_m) or abs(desired_offset_m)>.3:
            raise ValueError('intentional initial offset must be finite and at most 0.3 meters')
        if abs(reference.decision_time_s-state.timestamp_s) > self.config.control_dt_s*.1:
            raise ValueError("reference evidence decision timestamp differs from physical state")
        if not np.isfinite([state.x_m,state.y_m,state.yaw_rad,state.speed_mps,state.yaw_rate_rps,
                            state.timestamp_s,desired_speed_mps]).all() or state.speed_mps < 0:
            raise ValueError("invalid physical state or desired speed")
        if state.lateral_acceleration_mps2 is not None and not math.isfinite(state.lateral_acceleration_mps2):
            raise ValueError("measured lateral acceleration must be finite")
        if self._last_timestamp is not None and state.timestamp_s <= self._last_timestamp:
            raise ValueError("controller timestamps must increase; reset between episodes")
        c,p=self.config,self.calibration; n=c.horizon_steps; h=c.prediction_dt_s
        s0,_=reference.project([state.x_m,state.y_m])
        first_position,first_yaw,_,_=reference.sample(reference.s[0])
        before_first=float(np.dot(first_position[:2]-[state.x_m,state.y_m],
                                  [math.cos(first_yaw),math.sin(first_yaw)]))
        if s0-reference.s[0]<.05 and before_first>0:
            # Schedule arrival at the first visible goal from recorded pose.
            # Queries below the first knot remain fixed at that observed goal;
            # no unseen lane geometry or extrapolated spline enters the MPC.
            s0=float(reference.s[0]-before_first)
        self._last_command_rad = self._last_command_rad if self._history else self._initial_steering(state)
        if not self._history:self._previous_command_rad=self._last_command_rad
        # Constrain the driven trajectory's acceleration and jerk below, rather
        # than converting the largest spline curvature anywhere in the observed
        # corridor into an immediate cruise-speed step. Sparse paint and marking
        # transitions can create distant derivative spikes without requiring that
        # curvature in the physically feasible trajectory.
        stopping_limit = math.sqrt(max(0.,2*abs(c.min_acceleration_mps2)*(reference.s[-1]-s0-c.vehicle_length_m/2)))
        target_speed = max(0., min(desired_speed_mps,stopping_limit))
        # Predict only through observed road, retaining enough time to stop.
        # A fixed 2.4-second horizon can become infeasible at a paint gap even
        # when the complete immediate action and stopping corridor are observed.
        available=reference.s[-1]-s0-c.vehicle_length_m/2
        supported_steps=int(math.floor(available/max(state.speed_mps,target_speed,.1)/h))
        stopping_steps=max(4,int(math.ceil((p.delay_s+state.speed_mps/abs(c.min_acceleration_mps2))/h)))
        n=min(n,supported_steps)
        if n<stopping_steps:
            raise ValueError('insufficient_observed_stopping_corridor')
        sample_s = np.clip(s0+target_speed*np.arange(1,n+1)*h,reference.s[0],reference.s[-1])
        _,_,initial_curvature,_ = reference.sample(sample_s)
        initial = np.r_[np.arctan(p.wheelbase_m*initial_curvature),
                        np.full(n,np.clip((target_speed-state.speed_mps)/2,c.min_acceleration_mps2,c.max_acceleration_mps2))]
        if self._previous_solution is not None:
            old_n=len(self._previous_solution)//2
            times=np.arange(n)*h;old_times=np.arange(old_n)*h
            initial=np.r_[np.interp(times,old_times,self._previous_solution[:old_n]),
                          np.interp(times,old_times,self._previous_solution[old_n:])]
        delta_limit = min(p.max_steering_rad,abs(p.steering_gain_rad))
        delta_lower=max(-delta_limit,p.angle_from_steer(p.steering_fraction_min))
        delta_upper=min(delta_limit,p.angle_from_steer(p.steering_fraction_max))
        initial_ay=state.lateral_acceleration_mps2
        if initial_ay is None:
            initial_ay=math.sqrt(max(0.,state.speed_mps**2-(p.reference_offset_m*state.yaw_rate_rps)**2))*state.yaw_rate_rps
            if self._last_yaw_rate is not None:
                initial_ay+=p.reference_offset_m*(state.yaw_rate_rps-self._last_yaw_rate)/(state.timestamp_s-self._last_timestamp)
        cache = {}
        def terms(z):
            key=z.tobytes()
            if key in cache: return cache[key]
            dense=self._rollout(state,z,dense=True)
            subdivisions=max(1,int(round(h/c.control_dt_s)))
            roll=dense[::subdivisions]
            progress=s0+np.cumsum((roll[:-1,3]+roll[1:,3])*.5*h)
            q=np.clip(progress,reference.s[0],reference.s[-1])
            xyz,heading,kappa,width=reference.sample(q)
            delta_xy=roll[1:,:2]-xyz[:,:2]
            lateral=-np.sin(heading)*delta_xy[:,0]+np.cos(heading)*delta_xy[:,1]
            longitudinal=np.cos(heading)*delta_xy[:,0]+np.sin(heading)*delta_xy[:,1]
            body_heading_error=wrap_angle(roll[1:,2]-heading)
            beta=np.arcsin(np.clip(p.reference_offset_m*roll[1:,4]/np.maximum(roll[1:,3],.1),-.95,.95))
            heading_error=wrap_angle(roll[1:,2]+beta-heading)
            ay=reference_lateral_acceleration(dense[:,3],dense[:,4],h/subdivisions,p.reference_offset_m,initial_ay)
            jerk=np.diff(ay)/(h/subdivisions)
            # Oriented body support protects both front and rear corners.
            extent=.5*c.vehicle_width_m*np.abs(np.cos(body_heading_error))+.5*c.vehicle_length_m*np.abs(np.sin(body_heading_error))
            clearance=width*.5-extent-c.clearance_m
            command_rate=np.diff(np.r_[self._last_command_rad,z[:n]])/np.r_[c.control_dt_s,np.full(n-1,h)]
            command_ay=roll[1:,3]**2*np.tan(z[:n])/p.wheelbase_m
            # The real actuator receives a new first command every 50 ms, not
            # the optimizer's smooth 150 ms future ramp. Bound the jerk of those
            # executed command knots, including actor-reference yaw acceleration.
            previous_angle=self._history[-2][1] if len(self._history)>=2 else self._previous_command_rad
            last_k=math.tan(self._last_command_rad)/p.wheelbase_m
            previous_k=math.tan(previous_angle)/p.wheelbase_m
            command_rates=np.r_[state.speed_mps*previous_k,state.speed_mps*last_k,
                roll[:-1,3]*np.tan(z[:n])/p.wheelbase_m]
            command_intervals=np.r_[c.control_dt_s,c.control_dt_s,np.full(n-1,h)]
            command_lateral=np.r_[state.speed_mps,roll[:-1,3]]*command_rates[1:]+p.reference_offset_m*np.diff(command_rates)/command_intervals
            executed_command_jerk=np.diff(command_lateral)/command_intervals[1:]
            margins=np.r_[c.max_lateral_acceleration_mps2-np.abs(ay[1:]),
                c.max_lateral_acceleration_mps2-np.abs(command_ay),
                c.max_steering_rate_rps-np.abs(command_rate),
                clearance-np.abs(lateral),reference.s[-1]-progress-c.vehicle_length_m/2,
                c.max_executed_command_jerk_mps3-np.abs(executed_command_jerk)]
            if p.qualified and p.speed_min_mps<=state.speed_mps<=p.speed_max_mps:
                # A steady calibrated maneuver cannot borrow hypothetical
                # braking outside the identified actuator domain to satisfy
                # its lateral constraints. Startup stays diagnostic-only.
                margins=np.r_[margins,dense[1:,3]-p.speed_min_mps,p.speed_max_mps-dense[1:,3]]
            track_lateral=lateral-desired_offset_m
            cost=float(np.sum(24*track_lateral**2+2*longitudinal**2+14*heading_error**2+
                .7*(roll[1:,3]-target_speed)**2+.1*(z[:n]-np.arctan(p.wheelbase_m*kappa))**2)
                # The first difference crosses measured and model acceleration.
                # Terrain/suspension residuals there are not steering dynamics;
                # penalizing them feeds road disturbances back into steering.
                # Executed command jerk remains bounded at the current action.
                +.03*np.sum(jerk[1:]**2)/subdivisions
                +10.*np.sum(np.maximum(0.,np.abs(jerk[1:])-c.max_lateral_jerk_mps3)**2)
                +45*track_lateral[-1]**2+15*heading_error[-1]**2)
            result=(cost,margins,roll,progress)
            # SLSQP requests the same finite-difference stencil separately for
            # cost and constraints. Reuse identical results for this solve.
            if len(cache)>=4096:cache.clear()
            cache[key]=result
            return result
        solved=minimize(lambda z:terms(z)[0],initial,method="SLSQP",
            bounds=[(delta_lower,delta_upper)]*n+[(c.min_acceleration_mps2,c.max_acceleration_mps2)]*n,
            constraints=[{"type":"ineq","fun":lambda z:terms(z)[1]}],
            options={"maxiter":c.max_iterations,"ftol":1e-7,"disp":False})
        retried=False
        if not solved.success and solved.status in (8,9) and np.min(terms(solved.x)[1])>=-.01:
            # Near-feasible iteration/line-search failures are not accepted.
            # Continue with a less restrictive objective convergence tolerance;
            # the independent physical constraint tolerance remains 1e-4.
            solved=minimize(lambda z:terms(z)[0],solved.x,method="SLSQP",
                bounds=[(delta_lower,delta_upper)]*n+[(c.min_acceleration_mps2,c.max_acceleration_mps2)]*n,
                constraints=[{"type":"ineq","fun":lambda z:terms(z)[1]}],
                options={"maxiter":2*c.max_iterations,"ftol":1e-4,"disp":False})
            retried=True
        _,margins,predicted,progress=terms(solved.x)
        margin=float(np.min(margins)); reasons=[]
        subdivisions=max(1,int(round(h/c.control_dt_s)));dense_count=n*subdivisions
        lengths=[('predicted_lateral_acceleration',dense_count),('command_lateral_acceleration',n),
                 ('steering_rate',n),('body_clearance',n),
                 ('observed_body_support',n),('executed_command_jerk',n)]
        if len(margins)>sum(length for _,length in lengths):
            lengths.extend([('calibrated_speed_minimum',dense_count),('calibrated_speed_maximum',dense_count)])
        constraint_margins={};cursor=0
        for name,length in lengths:
            constraint_margins[name]=float(np.min(margins[cursor:cursor+length]));cursor+=length
        if not solved.success: reasons.append("mpc_optimization_failed")
        if margin < -1e-4: reasons.append("mpc_constraints_violated")
        if c.require_calibration and not (p.speed_min_mps-c.speed_calibration_margin_mps <= state.speed_mps <= p.speed_max_mps+c.speed_calibration_margin_mps):
            reasons.append("speed_outside_steering_calibration")
        delta=float(solved.x[0]); curvature_command=math.tan(delta)/p.wheelbase_m
        carla_steer=p.steer_from_angle(delta)
        if c.require_calibration and not (p.steering_fraction_min-1e-4 <= carla_steer <= p.steering_fraction_max+1e-4):
            reasons.append("command_outside_measured_steering_range")
        # Spline coefficients depend on every admitted knot, so attribution
        # must include the full observed input.
        evidence=reference.evidence_for_arc(reference.s[0],reference.s[-1])
        accepted=not reasons
        if accepted:
            self._history.append((state.timestamp_s,delta))
            self._history=[item for item in self._history if item[0]>=state.timestamp_s-p.delay_s-1.]
            self._previous_command_rad=self._last_command_rad
            self._last_command_rad=delta;self._previous_solution=solved.x.copy()
        self._last_timestamp=state.timestamp_s;self._last_observed_delta=self._initial_steering(state)
        self._last_yaw_rate=state.yaw_rate_rps
        return MPCCommand(accepted,tuple(reasons),curvature_command,carla_steer,
            delta,float(solved.x[n]),target_speed,p.delay_s+c.control_dt_s,predicted,evidence,
            bool(solved.success),str(solved.message)+(" (after bounded solver retry)" if retried else ""),margin,constraint_margins)
