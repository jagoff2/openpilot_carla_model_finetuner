"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos/
"""


# Runtime source excerpts retained verbatim for AST-based contract tests.
def get_action_from_model(outputs: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          v_ego: float, lat_action_t: float, long_action_t: float,
                          lat_smooth_seconds: float | None = None) -> log.ModelDataV2.Action:
  if "action" in outputs:
    desired_accel = float(outputs["action"][0, 1])
    desired_curvature = float(outputs["action"][0, 0]) / (max(1.0, v_ego)) ** 2
    should_stop = bool(v_ego < 0.3 and desired_accel < 0.1)
  else:
    from iqpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan
    from iqpilot.selfdrive.iqmodeld.config import ModelConstants, Plan
    plan = outputs["plan"][0]
    desired_accel, should_stop = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0],
                                                     plan[:, Plan.ACCELERATION][:, 0],
                                                     ModelConstants.T_IDXS,
                                                     action_t=long_action_t)
    desired_curvature = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                                                plan[:, Plan.ORIENTATION_RATE][:, 2],
                                                ModelConstants.T_IDXS, v_ego, lat_action_t)
    desired_accel, should_stop = float(desired_accel), bool(should_stop)
    desired_curvature = float(desired_curvature)
  desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)
  if v_ego > MIN_LAT_CONTROL_SPEED:
    lat_smooth = LAT_SMOOTH_SECONDS if lat_smooth_seconds is None else lat_smooth_seconds
    desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, lat_smooth)
  else:
    desired_curvature = prev_action.desiredCurvature
  return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                desiredAcceleration=float(desired_accel),
                                shouldStop=should_stop,
                                lateralActionSource="nativeAction" if "action" in outputs else "plan")
