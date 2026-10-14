"""Physical labels and independently masked means for the 2,580-output ABI."""
from __future__ import annotations

import numpy as np

OUTPUT_WIDTH = 2580
PLAN_MEAN_SLICE = slice(917, 1412)
LATERAL_MEAN_INDEX = 2062
LONGITUDINAL_MEAN_INDEX = 2063
PLAN_TIMES = 10.0 * (np.arange(33, dtype=np.float64) / 32.0) ** 2


def lateral_action_target(current_speed, commanded_curvature):
    """Encode the expert command at the recorded lateral action horizon.

    Caller supplies the *commanded* curvature, not realized yaw_rate/speed or a
    curvature sampled at future speed. Positive curvature is FRD positive yaw.
    """
    speed, curvature = np.broadcast_arrays(np.asarray(current_speed, dtype=np.float64),
                                          np.asarray(commanded_curvature, dtype=np.float64))
    if not np.isfinite(speed).all() or not np.isfinite(curvature).all() or np.any(speed < 0):
        raise ValueError("speed and commanded curvature must be finite; speed nonnegative")
    return np.maximum(1.0, speed) ** 2 * curvature


def decode_lateral_action(target, current_speed):
    return np.asarray(target) / np.maximum(1.0, np.asarray(current_speed)) ** 2


def _carla_rotations(rpy):
    """CARLA Transform.get_matrix rotation, reflected into right-handed FRD."""
    from scipy.spatial.transform import Rotation
    # CARLA rotation equals Rz(yaw) Ry(-pitch) Rx(-roll) in its x,y,z coordinates.
    # Reflect world and vehicle z, obtaining Rz(yaw) Ry(pitch) Rx(roll).
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64))


def dense_to_plan(timestamps, position_world, velocity_world, rotation_world,
                  label_indices, observable_support, *, max_interpolation_gap=0.05001):
    """Return coherent current-relative FRD [L,33,15] plans and element masks.

    World positions/velocities use CARLA x,y,z; rotations are CARLA roll,pitch,
    yaw radians. A single Hermite position interpolant produces position,
    velocity and acceleration. SLERP produces relative orientation; its segment
    angular velocity is expressed in the *current* FRD frame. Unsupported and
    unavailable future intervals remain masked, including intervals across a
    missing sensor tick. The initial position and orientation are exactly zero.
    """
    from scipy.interpolate import CubicHermiteSpline
    from scipy.spatial.transform import Slerp
    time = np.asarray(timestamps, dtype=np.float64)
    pos = np.asarray(position_world, dtype=np.float64)
    vel = np.asarray(velocity_world, dtype=np.float64)
    rpy = np.asarray(rotation_world, dtype=np.float64)
    indices = np.asarray(label_indices)
    support = np.asarray(observable_support)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if any(a.shape != (len(time), 3) for a in (pos, vel, rpy)):
        raise ValueError("physical arrays must have shape [T,3]")
    if not all(np.isfinite(a).all() for a in (time, pos, vel, rpy)):
        raise ValueError("physical arrays must be finite")
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(time)):
        raise ValueError("label_indices must be in-range integer indices")
    if support.dtype != np.bool_ or support.shape not in ((len(indices), 33), (len(indices), 33, 15)):
        raise ValueError("observable_support must be bool [L,33] or [L,33,15]")
    reflection = np.array([1.0, 1.0, -1.0])
    spline = CubicHermiteSpline(time, pos * reflection, vel * reflection, extrapolate=False)
    orientations = _carla_rotations(rpy)
    slerp = Slerp(time, orientations)
    segment_dt = np.diff(time)
    # Relative rotvec is body-local; transform each segment rate into world FRD.
    segment_rates = orientations[:-1].apply((orientations[:-1].inv() * orientations[1:]).as_rotvec()
                                             / segment_dt[:, None])
    plans = np.zeros((len(indices), 33, 15), np.float32)
    masks = np.zeros_like(plans, dtype=bool)
    for row, index in enumerate(indices):
        query = time[index] + PLAN_TIMES
        in_range = query <= time[-1] + 1e-9
        query = np.clip(query, time[0], time[-1])
        segments = np.minimum(np.searchsorted(time, query, side="right") - 1, len(time) - 2)
        available = in_range & (segment_dt[segments] <= max_interpolation_gap)
        current_inv = orientations[index].inv()
        plans[row, :, 0:3] = current_inv.apply(spline(query) - pos[index] * reflection)
        plans[row, :, 3:6] = current_inv.apply(spline(query, 1))
        plans[row, :, 6:9] = current_inv.apply(spline(query, 2))
        relative = current_inv * slerp(query)
        # Unwrap each angle through pi so a hairpin does not jump at the branch cut.
        plans[row, :, 9:12] = np.unwrap(relative.as_euler("xyz"), axis=0)
        plans[row, :, 12:15] = current_inv.apply(segment_rates[segments])
        row_support = support[row, :, None] if support.ndim == 2 else support[row]
        masks[row] = available[:, None] & row_support
        plans[row, 0, 0:3] = 0.0
        plans[row, 0, 9:12] = 0.0
    return plans, masks


def expert_trajectory_loss(predictions, plan_targets=None, plan_mask=None,
                           lateral_targets=None, lateral_mask=None,
                           longitudinal_targets=None, longitudinal_mask=None,
                           candidate_mask=None, valid_tick_mask=None,
                           plan_scales=None, lateral_scale=2.5, longitudinal_scale=2.0):
    """Set-valued masked SmoothL1 loss with no fabricated action labels.

    Targets use [T,K,33,15] plans or [T,K] actions; a singleton candidate axis
    may be omitted. Each available head has equal weight *per candidate*.
    A single candidate is chosen jointly across its available heads. Only
    supported mean outputs receive gradients. All-masked batches return an
    exact differentiable zero even when masked values are NaN.
    """
    import torch
    import torch.nn.functional as F
    if not isinstance(predictions, torch.Tensor) or predictions.ndim != 2 or predictions.shape[1] != OUTPUT_WIDTH:
        raise ValueError("predictions must be a tensor [T,2580]")
    if not predictions.is_floating_point():
        raise TypeError("predictions must be floating point")
    ticks = predictions.shape[0]
    heads = []
    for name, target, mask, selection, scale in (
        ("plan", plan_targets, plan_mask, PLAN_MEAN_SLICE,
         [1.0] * 15 if plan_scales is None else plan_scales),
        ("lateral", lateral_targets, lateral_mask, LATERAL_MEAN_INDEX, lateral_scale),
        ("longitudinal", longitudinal_targets, longitudinal_mask, LONGITUDINAL_MEAN_INDEX, longitudinal_scale),
    ):
        if target is None and mask is None:
            continue
        if target is None or mask is None:
            raise ValueError(f"{name} target and mask must be supplied together")
        if not isinstance(target, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("targets and masks must be tensors")
        if mask.dtype != torch.bool or not target.is_floating_point():
            raise TypeError("targets must be floating point and masks torch.bool")
        if target.device != predictions.device or mask.device != predictions.device:
            raise ValueError("targets and masks must share prediction device")
        tail = (33, 15) if name == "plan" else ()
        if target.ndim == 1 + len(tail):
            target, mask = target[:, None], mask[:, None]
        if target.ndim != 2 + len(tail) or target.shape[0] != ticks or tuple(target.shape[2:]) != tail or target.shape != mask.shape:
            raise ValueError(f"invalid {name} target/mask shape")
        scales = torch.as_tensor(scale, dtype=predictions.dtype, device=predictions.device)
        if (name == "plan" and scales.shape != (15,)) or (name != "plan" and scales.ndim):
            raise ValueError(f"invalid {name} scale shape")
        if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
            raise ValueError("scales must be positive finite")
        heads.append((name, target, mask, selection, scales, tail))
    if not heads:
        raise ValueError("at least one target head is required")
    candidates = heads[0][1].shape[1]
    if candidates < 1 or any(h[1].shape[1] != candidates for h in heads):
        raise ValueError("all heads must have the same nonempty candidate axis")
    def checked_mask(value, shape, name):
        if value is None:
            return torch.ones(shape, dtype=torch.bool, device=predictions.device)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or tuple(value.shape) != shape or value.device != predictions.device:
            raise ValueError(f"invalid {name}")
        return value
    candidates_ok = checked_mask(candidate_mask, (ticks, candidates), "candidate_mask")
    ticks_ok = checked_mask(valid_tick_mask, (ticks,), "valid_tick_mask")
    active = candidates_ok & ticks_ok[:, None]
    total = predictions.new_zeros((ticks, candidates))
    counts = predictions.new_zeros((ticks, candidates))
    diagnostics = {}
    for name, target, mask, selection, scales, tail in heads:
        admitted = mask & active.reshape(ticks, candidates, *([1] * len(tail)))
        raw = predictions[:, selection].reshape(ticks, *tail)
        raw = raw[:, None].expand(ticks, candidates, *tail)
        if bool((admitted & (~torch.isfinite(target) | ~torch.isfinite(raw))).any()):
            raise ValueError(f"supported {name} targets/predictions must be finite")
        safe_pred = torch.where(admitted, raw, torch.zeros_like(raw))
        safe_target = torch.where(admitted, target, torch.zeros_like(target))
        element = F.smooth_l1_loss(safe_pred / scales, safe_target / scales, reduction="none")
        if tail:
            n = admitted.sum(dim=(-1, -2))
            value = element.sum(dim=(-1, -2)) / n.clamp_min(1)
        else:
            n, value = admitted, element
        total = total + value
        counts = counts + (n > 0).to(predictions.dtype)
        diagnostics[f"{name}_candidate_loss"] = value.detach()
    eligible = active & (counts > 0)
    valid = eligible.any(dim=1)
    combined = total / counts.clamp_min(1)
    chosen = combined.masked_fill(~eligible, torch.inf).argmin(dim=1)
    if bool(valid.any()):
        loss = combined[torch.arange(ticks, device=predictions.device), chosen][valid].mean()
    else:
        # Only admitted outputs participate. `predictions.sum()*0` leaks NaNs.
        loss = total.sum()
    diagnostics.update(chosen_candidate_indices=chosen[valid].detach(),
                       eligible_candidate_mask=eligible.detach(),
                       valid_tick_count=int(valid.sum()))
    return loss, diagnostics
