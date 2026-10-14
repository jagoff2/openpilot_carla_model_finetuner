"""
Dynamic bicycle model from "The Science of Vehicle Dynamics (2014), M. Guiggiani"

The state is x = [v, r]^T
with v lateral speed [m/s], and r rotational speed [rad/s]

The input u is the steering angle [rad], and roll [rad]

The system is defined by
x_dot = A*x + B*u

A depends on longitudinal speed, u [m/s], and vehicle parameters CP
"""


# Runtime source excerpts retained verbatim for AST-based contract tests.
class VehicleModel:
  def __init__(self, CP: CarParams):
    """
    Args:
      CP: Car Parameters
    """
    # for math readability, convert long names car params into short names
    self.m: float = CP.mass
    self.j: float = CP.rotationalInertia
    self.l: float = CP.wheelbase
    self.aF: float = CP.centerToFront
    self.aR: float = CP.wheelbase - CP.centerToFront
    self.chi: float = CP.steerRatioRear

    self.cF_orig: float = CP.tireStiffnessFront
    self.cR_orig: float = CP.tireStiffnessRear
    self.update_params(1.0, CP.steerRatio)

  def update_params(self, stiffness_factor: float, steer_ratio: float) -> None:
    """Update the vehicle model with a new stiffness factor and steer ratio"""
    self.cF: float = stiffness_factor * self.cF_orig
    self.cR: float = stiffness_factor * self.cR_orig
    self.sR: float = steer_ratio

  def steady_state_sol(self, sa: float, u: float, roll: float) -> np.ndarray:
    """Returns the steady state solution.

    If the speed is too low we can't use the dynamic model (tire slip is undefined),
    we then have to use the kinematic model

    Args:
      sa: Steering wheel angle [rad]
      u: Speed [m/s]
      roll: Road Roll [rad]

    Returns:
      2x1 matrix with steady state solution (lateral speed, rotational speed)
    """
    if u > 0.1:
      return dyn_ss_sol(sa, u, roll, self)
    else:
      return kin_ss_sol(sa, u, self)

  def calc_curvature(self, sa: float, u: float, roll: float) -> float:
    """Returns the curvature. Multiplied by the speed this will give the yaw rate.

    Args:
      sa: Steering wheel angle [rad]
      u: Speed [m/s]
      roll: Road Roll [rad]

    Returns:
      Curvature factor [1/m]
    """
    return (self.curvature_factor(u) * sa / self.sR) + self.roll_compensation(roll, u)

  def curvature_factor(self, u: float) -> float:
    """Returns the curvature factor.
    Multiplied by wheel angle (not steering wheel angle) this will give the curvature.

    Args:
      u: Speed [m/s]

    Returns:
      Curvature factor [1/m]
    """
    sf = calc_slip_factor(self)
    return (1. - self.chi) / (1. - sf * u**2) / self.l

  def get_steer_from_curvature(self, curv: float, u: float, roll: float) -> float:
    """Calculates the required steering wheel angle for a given curvature

    Args:
      curv: Desired curvature [1/m]
      u: Speed [m/s]
      roll: Road Roll [rad]

    Returns:
      Steering wheel angle [rad]
    """

    return (curv - self.roll_compensation(roll, u)) * self.sR * 1.0 / self.curvature_factor(u)

  def roll_compensation(self, roll: float, u: float) -> float:
    """Calculates the roll-compensation to curvature

    Args:
      roll: Road Roll [rad]
      u: Speed [m/s]

    Returns:
      Roll compensation curvature [rad]
    """
    sf = calc_slip_factor(self)

    if abs(sf) < 1e-6:
      return 0
    else:
      return (ACCELERATION_DUE_TO_GRAVITY * roll) / ((1 / sf) - u**2)

  def get_steer_from_yaw_rate(self, yaw_rate: float, u: float, roll: float) -> float:
    """Calculates the required steering wheel angle for a given yaw_rate

    Args:
      yaw_rate: Desired yaw rate [rad/s]
      u: Speed [m/s]
      roll: Road Roll [rad]

    Returns:
      Steering wheel angle [rad]
    """
    curv = yaw_rate / u
    return self.get_steer_from_curvature(curv, u, roll)

  def yaw_rate(self, sa: float, u: float, roll: float) -> float:
    """Calculate yaw rate

    Args:
      sa: Steering wheel angle [rad]
      u: Speed [m/s]
      roll: Road Roll [rad]

    Returns:
      Yaw rate [rad/s]
    """
    return self.calc_curvature(sa, u, roll) * u


def calc_slip_factor(VM: VehicleModel) -> float:
  """The slip factor is a measure of how the curvature changes with speed
  it's positive for Oversteering vehicle, negative (usual case) otherwise.
  """
  return VM.m * (VM.cF * VM.aF - VM.cR * VM.aR) / (VM.l**2 * VM.cF * VM.cR)
