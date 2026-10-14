"""Differentiable set-valued loss for qualified expert trajectories."""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


PLAN_MEAN_SLICE = slice(917, 1412)
ACCELERATION_MEAN_INDEX = 2063
OUTPUT_WIDTH = 2580


def _require_tensor(name: str, value: object) -> torch.Tensor:
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"{name} must be a torch.Tensor")
  return value


def _require_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
  if tuple(value.shape) != shape:
    raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def expert_trajectory_loss(
  predictions: torch.Tensor,
  plan_targets: torch.Tensor,
  plan_mask: torch.Tensor,
  acceleration_targets: torch.Tensor,
  acceleration_mask: torch.Tensor,
  candidate_mask: torch.Tensor,
  valid_tick_mask: torch.Tensor,
  plan_scales: torch.Tensor | Sequence[float],
  acceleration_scale: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
  """Choose the lowest-loss qualified trajectory independently at each tick.

  Residuals are divided by physical scales before applying Smooth L1 with
  ``beta=1``. Each head contributes one half of the combined candidate loss.
  Diagnostics are detached and therefore cannot extend the autograd graph.
  """
  predictions = _require_tensor("predictions", predictions)
  plan_targets = _require_tensor("plan_targets", plan_targets)
  plan_mask = _require_tensor("plan_mask", plan_mask)
  acceleration_targets = _require_tensor(
    "acceleration_targets", acceleration_targets)
  acceleration_mask = _require_tensor("acceleration_mask", acceleration_mask)
  candidate_mask = _require_tensor("candidate_mask", candidate_mask)
  valid_tick_mask = _require_tensor("valid_tick_mask", valid_tick_mask)

  if predictions.ndim != 2 or predictions.shape[1] != OUTPUT_WIDTH:
    raise ValueError(
      f"predictions must have shape [T, {OUTPUT_WIDTH}], got {tuple(predictions.shape)}")
  if not predictions.is_floating_point():
    raise TypeError("predictions must have a floating dtype")
  tick_count = predictions.shape[0]
  if plan_targets.ndim != 4 or plan_targets.shape[0] != tick_count or plan_targets.shape[2:] != (33, 15):
    raise ValueError("plan_targets must have shape [T, K, 33, 15]")
  candidate_count = plan_targets.shape[1]
  _require_shape("plan_mask", plan_mask, tuple(plan_targets.shape))
  _require_shape("acceleration_targets", acceleration_targets, (tick_count, candidate_count))
  _require_shape("acceleration_mask", acceleration_mask, (tick_count, candidate_count))
  _require_shape("candidate_mask", candidate_mask, (tick_count, candidate_count))
  _require_shape("valid_tick_mask", valid_tick_mask, (tick_count,))
  if not plan_targets.is_floating_point() or not acceleration_targets.is_floating_point():
    raise TypeError("plan_targets and acceleration_targets must have floating dtypes")
  for name, value in (("plan_mask", plan_mask),
                      ("acceleration_mask", acceleration_mask),
                      ("candidate_mask", candidate_mask),
                      ("valid_tick_mask", valid_tick_mask)):
    if value.dtype != torch.bool:
      raise TypeError(f"{name} must have dtype torch.bool")
  for name, value in (("plan_targets", plan_targets), ("plan_mask", plan_mask),
                      ("acceleration_targets", acceleration_targets),
                      ("acceleration_mask", acceleration_mask),
                      ("candidate_mask", candidate_mask),
                      ("valid_tick_mask", valid_tick_mask)):
    if value.device != predictions.device:
      raise ValueError(f"{name} must be on predictions.device")

  scales = torch.as_tensor(plan_scales, device=predictions.device,
                           dtype=predictions.dtype)
  _require_shape("plan_scales", scales, (15,))
  if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
    raise ValueError("plan_scales must contain 15 positive finite values")
  try:
    acceleration_scale_value = float(acceleration_scale)
  except (TypeError, ValueError) as exc:
    raise TypeError("acceleration_scale must be a real scalar") from exc
  scale_tensor = predictions.new_tensor(acceleration_scale_value)
  if not bool(torch.isfinite(scale_tensor)) or acceleration_scale_value <= 0:
    raise ValueError("acceleration_scale must be positive and finite")
  if not bool(valid_tick_mask.any()):
    raise ValueError("at least one valid tick is required")

  plan_counts = plan_mask.sum(dim=(-1, -2))
  active = candidate_mask & valid_tick_mask[:, None]
  missing_plan = active & (plan_counts == 0)
  missing_acceleration = active & ~acceleration_mask
  if bool(missing_plan.any()):
    raise ValueError("every declared active candidate must have an accepted plan element")
  if bool(missing_acceleration.any()):
    raise ValueError("every declared active candidate must have an accepted acceleration element")
  eligible = active
  if bool((valid_tick_mask & ~eligible.any(dim=1)).any()):
    raise ValueError("every valid tick must have at least one eligible candidate")

  eligible_plan_values = eligible[:, :, None, None] & plan_mask
  if bool((~torch.isfinite(plan_targets) & eligible_plan_values).any()):
    raise ValueError("eligible plan target values must be finite")
  if bool((~torch.isfinite(acceleration_targets) & eligible).any()):
    raise ValueError("eligible acceleration target values must be finite")
  selected_predictions = torch.cat(
    (predictions[:, PLAN_MEAN_SLICE],
     predictions[:, ACCELERATION_MEAN_INDEX:ACCELERATION_MEAN_INDEX + 1]), dim=1)
  if bool((~torch.isfinite(selected_predictions[valid_tick_mask])).any()):
    raise ValueError("selected prediction values at valid ticks must be finite")

  raw_plan_prediction = predictions[:, PLAN_MEAN_SLICE].reshape(tick_count, 33, 15)
  plan_prediction = torch.where(
    valid_tick_mask[:, None, None], raw_plan_prediction, torch.zeros_like(raw_plan_prediction))
  safe_plan_targets = torch.where(eligible_plan_values, plan_targets,
                                  plan_prediction[:, None])
  plan_element_loss = F.smooth_l1_loss(
    plan_prediction[:, None].expand_as(safe_plan_targets) / scales,
    safe_plan_targets / scales,
    beta=1.0,
    reduction="none",
  )
  plan_losses = (
    torch.where(eligible_plan_values, plan_element_loss,
                torch.zeros((), device=predictions.device, dtype=predictions.dtype))
    .sum(dim=(-1, -2)) / plan_counts.clamp_min(1).to(predictions.dtype)
  )

  raw_acceleration_prediction = predictions[:, ACCELERATION_MEAN_INDEX, None]
  acceleration_prediction = torch.where(
    valid_tick_mask[:, None], raw_acceleration_prediction,
    torch.zeros_like(raw_acceleration_prediction))
  safe_acceleration_targets = torch.where(
    eligible, acceleration_targets, acceleration_prediction)
  acceleration_losses = F.smooth_l1_loss(
    acceleration_prediction.expand_as(safe_acceleration_targets) / acceleration_scale_value,
    safe_acceleration_targets / acceleration_scale_value,
    beta=1.0,
    reduction="none",
  )
  combined_losses = 0.5 * plan_losses + 0.5 * acceleration_losses
  masked_combined = combined_losses.masked_fill(~eligible, torch.inf)
  chosen_indices = torch.argmin(masked_combined, dim=1)
  rows = torch.arange(tick_count, device=predictions.device)
  chosen_plan = plan_losses[rows, chosen_indices]
  chosen_acceleration = acceleration_losses[rows, chosen_indices]
  chosen_combined = combined_losses[rows, chosen_indices]
  loss = chosen_combined[valid_tick_mask].mean()

  diagnostics: dict[str, Any] = {
    "chosen_candidate_indices": chosen_indices[valid_tick_mask].detach(),
    "chosen_plan_loss": chosen_plan[valid_tick_mask].detach(),
    "chosen_acceleration_loss": chosen_acceleration[valid_tick_mask].detach(),
    "chosen_combined_loss": chosen_combined[valid_tick_mask].detach(),
    "eligible_candidate_mask": eligible.detach(),
    "valid_tick_count": int(valid_tick_mask.sum().item()),
  }
  return loss, diagnostics


def expert_acceleration_loss(
  predictions: torch.Tensor,
  acceleration_targets: torch.Tensor,
  candidate_mask: torch.Tensor,
  valid_tick_mask: torch.Tensor,
  acceleration_scale: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
  """Set-valued loss for raw output 2063 longitudinal acceleration only."""
  predictions = _require_tensor("predictions", predictions)
  acceleration_targets = _require_tensor(
    "acceleration_targets", acceleration_targets)
  candidate_mask = _require_tensor("candidate_mask", candidate_mask)
  valid_tick_mask = _require_tensor("valid_tick_mask", valid_tick_mask)

  if predictions.ndim != 2 or predictions.shape[1] != OUTPUT_WIDTH:
    raise ValueError(
      f"predictions must have shape [T, {OUTPUT_WIDTH}], got {tuple(predictions.shape)}")
  if not predictions.is_floating_point():
    raise TypeError("predictions must have a floating dtype")
  tick_count = predictions.shape[0]
  if acceleration_targets.ndim != 2 or acceleration_targets.shape[0] != tick_count:
    raise ValueError("acceleration_targets must have shape [T, K]")
  if not acceleration_targets.is_floating_point():
    raise TypeError("acceleration_targets must have a floating dtype")
  candidate_count = acceleration_targets.shape[1]
  _require_shape("candidate_mask", candidate_mask, (tick_count, candidate_count))
  _require_shape("valid_tick_mask", valid_tick_mask, (tick_count,))
  if candidate_mask.dtype != torch.bool:
    raise TypeError("candidate_mask must have dtype torch.bool")
  if valid_tick_mask.dtype != torch.bool:
    raise TypeError("valid_tick_mask must have dtype torch.bool")
  for name, value in (("acceleration_targets", acceleration_targets),
                      ("candidate_mask", candidate_mask),
                      ("valid_tick_mask", valid_tick_mask)):
    if value.device != predictions.device:
      raise ValueError(f"{name} must be on predictions.device")
  try:
    acceleration_scale_value = float(acceleration_scale)
  except (TypeError, ValueError) as exc:
    raise TypeError("acceleration_scale must be a real scalar") from exc
  scale_tensor = predictions.new_tensor(acceleration_scale_value)
  if not bool(torch.isfinite(scale_tensor)) or acceleration_scale_value <= 0:
    raise ValueError("acceleration_scale must be positive and finite")
  if not bool(valid_tick_mask.any()):
    raise ValueError("at least one valid tick is required")

  active = candidate_mask & valid_tick_mask[:, None]
  if bool((valid_tick_mask & ~active.any(dim=1)).any()):
    raise ValueError("every valid tick must have at least one candidate")
  if bool((~torch.isfinite(acceleration_targets) & active).any()):
    raise ValueError("active acceleration target values must be finite")
  raw_prediction = predictions[:, ACCELERATION_MEAN_INDEX, None]
  if bool((~torch.isfinite(raw_prediction[valid_tick_mask])).any()):
    raise ValueError("acceleration predictions at valid ticks must be finite")
  prediction = torch.where(
    valid_tick_mask[:, None], raw_prediction, torch.zeros_like(raw_prediction))
  safe_targets = torch.where(active, acceleration_targets, prediction)
  candidate_losses = F.smooth_l1_loss(
    prediction.expand_as(safe_targets) / acceleration_scale_value,
    safe_targets / acceleration_scale_value,
    beta=1.0,
    reduction="none",
  )
  masked_losses = candidate_losses.masked_fill(~active, torch.inf)
  chosen_indices = torch.argmin(masked_losses, dim=1)
  rows = torch.arange(tick_count, device=predictions.device)
  chosen_losses = candidate_losses[rows, chosen_indices]
  loss = chosen_losses[valid_tick_mask].mean()
  diagnostics: dict[str, Any] = {
    "chosen_candidate_indices": chosen_indices[valid_tick_mask].detach(),
    "chosen_acceleration_loss": chosen_losses[valid_tick_mask].detach(),
    "active_candidate_mask": active.detach(),
    "valid_tick_count": int(valid_tick_mask.sum().item()),
  }
  return loss, diagnostics


__all__ = ["expert_acceleration_loss", "expert_trajectory_loss"]
