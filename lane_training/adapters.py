"""Explicit output-layout adapters; none selects a base checkpoint."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanPrediction:
    """Current-relative forward/right XY metres and optional yaw radians."""
    xy: torch.Tensor
    yaw: torch.Tensor | None = None


@dataclass(frozen=True)
class FlatPlanAdapter:
    """Decode a documented flat output layout, without assuming model identity."""
    output_width: int
    plan_start: int
    point_count: int
    point_width: int
    x_column: int = 0
    y_column: int = 1
    yaw_column: int | None = None

    def __post_init__(self):
        columns = [self.x_column, self.y_column]
        if self.yaw_column is not None:
            columns.append(self.yaw_column)
        if (self.point_count < 2 or self.point_width < 2 or self.plan_start < 0
                or self.plan_start+self.point_count*self.point_width > self.output_width
                or min(columns) < 0 or max(columns) >= self.point_width
                or len(columns) != len(set(columns))):
            raise ValueError("invalid plan output layout")

    def __call__(self, output):
        if output.ndim != 2 or output.shape[1] != self.output_width or not output.is_floating_point():
            raise ValueError("model output does not match the explicit plan adapter")
        plan = output[:, self.plan_start:self.plan_start+self.point_count*self.point_width]
        plan = plan.reshape(-1, self.point_count, self.point_width)
        return PlanPrediction(plan[:, :, [self.x_column, self.y_column]],
            None if self.yaw_column is None else plan[:, :, self.yaw_column])


def model_centerline_loss(predictions, reference_predictions, corridors, *, adapter, config=None):
    from .centerline import centerline_loss
    prediction = adapter(predictions)
    reference = adapter(reference_predictions.detach())
    return centerline_loss(prediction.xy, reference.xy, corridors,
        predicted_yaw=prediction.yaw, config=config)
