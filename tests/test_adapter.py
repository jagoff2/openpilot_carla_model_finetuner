import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lane_dataset.adapter import (PLAN_TIMES, dense_to_plan, expert_trajectory_loss,
                                  lateral_action_target)
from lane_dataset.preprocessing import DEPENDENCY_ROOT, StockPreprocessor, pack_yuv420, image_to_nv12, split_nv12


def load_actual_action_decoder():
    root = DEPENDENCY_ROOT / "runtime"
    namespace = {"np": np, "DT_MDL": 0.05, "LAT_SMOOTH_SECONDS": 0.0,
                 "LONG_SMOOTH_SECONDS": 0.3, "MIN_LAT_CONTROL_SPEED": 0.3,
                 "log": SimpleNamespace(ModelDataV2=SimpleNamespace(Action=lambda **kwargs: SimpleNamespace(**kwargs)))}
    for path, name in ((root / "drive_helpers.py", "smooth_value"),
                       (root / "driving_action.py", "get_action_from_model")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        assert len(nodes) == 1
        code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace["get_action_from_model"]


@pytest.mark.parametrize("speed", [0.31, 0.8, 1.0, 7.3, 15.0])
@pytest.mark.parametrize("curvature", [-0.05, 0.0, 0.05])
def test_lateral_target_through_actual_decoder(speed, curvature):
    actual = load_actual_action_decoder()
    previous = SimpleNamespace(desiredAcceleration=0.0, desiredCurvature=0.0)
    encoded = lateral_action_target(speed, curvature)
    result = actual({"action": np.array([[encoded, 0.0]])}, previous, speed, 0.25, 0.35)
    assert result.desiredCurvature == pytest.approx(curvature)
    assert result.lateralActionSource == "nativeAction"


def test_actual_decoder_low_speed_holds_previous_curvature():
    actual = load_actual_action_decoder()
    previous = SimpleNamespace(desiredAcceleration=0.0, desiredCurvature=-0.01)
    result = actual({"action": np.array([[0.03, 0]])}, previous, 0.2, 0.25, 0.35)
    assert result.desiredCurvature == -0.01


@pytest.mark.parametrize("curvature", [-0.04, 0, 0.04])
@pytest.mark.parametrize("roll", [0.0, 0.08])
def test_actual_runtime_angle_controller_and_vehicle_model_roundtrip(curvature, roll):
    """Execute source methods, with message plumbing and sedan parameters as fixtures."""
    iqroot = DEPENDENCY_ROOT / "runtime"
    model_source = iqroot / "vehicle_model.py"
    controller_source = iqroot / "latcontrol_angle.py"
    namespace = {"np": np, "CarParams": object, "ACCELERATION_DUE_TO_GRAVITY": 9.81,
                 "math": math, "STEER_ANGLE_SATURATION_THRESHOLD": 2.5,
                 "log": SimpleNamespace(ControlsState=SimpleNamespace(LateralAngleState=SimpleNamespace(new_message=SimpleNamespace)))}
    tree = ast.parse(model_source.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in ("VehicleModel", "calc_slip_factor")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(model_source), "exec"), namespace)
    tree = ast.parse(controller_source.read_text(encoding="utf-8-sig"))
    controller = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LatControlAngle")
    update = next(node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name == "update")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[update], type_ignores=[])), str(controller_source), "exec"), namespace)
    parameters = SimpleNamespace(mass=1600, rotationalInertia=2500, wheelbase=2.8,
        centerToFront=1.2, steerRatioRear=0, tireStiffnessFront=90000,
        tireStiffnessRear=95000, steerRatio=15)
    vm = namespace["VehicleModel"](parameters)
    current_speed = 7.0
    actual = load_actual_action_decoder()
    previous = SimpleNamespace(desiredAcceleration=0.0, desiredCurvature=0.0)
    decoded = actual({"action": np.array([[lateral_action_target(current_speed, curvature), 0]])},
                     previous, current_speed, .25, .35)
    controller_state = SimpleNamespace(use_steer_limited_by_safety=False,
                                       _check_saturation=lambda *args: bool(args[0]))
    params = SimpleNamespace(roll=roll, angleOffsetDeg=2)
    _, angle, _ = namespace["update"](controller_state, True,
        SimpleNamespace(steeringAngleDeg=2, vEgo=current_speed), vm, params, False,
        decoded.desiredCurvature, None, False, .25)
    reconstructed_frd_curvature = -vm.calc_curvature(math.radians(angle - params.angleOffsetDeg), current_speed, roll)
    assert reconstructed_frd_curvature == pytest.approx(curvature)


def test_lateral_only_no_longitudinal_or_uncertainty_gradient():
    prediction = torch.zeros((2, 2580), requires_grad=True)
    prediction.data[1] = float("nan")
    targets = torch.tensor([2.5, float("nan")])
    loss, info = expert_trajectory_loss(prediction, lateral_targets=targets,
        lateral_mask=torch.tensor([True, False]))
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert torch.nonzero(prediction.grad).tolist() == [[0, 2062]]
    assert info["valid_tick_count"] == 1


def test_unsupported_values_produce_exact_zero_loss_and_gradient():
    prediction = torch.full((2, 2580), float("nan"), requires_grad=True)
    plan = torch.full((2, 33, 15), float("nan"))
    loss, info = expert_trajectory_loss(prediction, plan, torch.zeros_like(plan, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert info["valid_tick_count"] == 0


def test_joint_candidate_selection_and_independent_head_masks():
    prediction = torch.zeros((1, 2580), requires_grad=True)
    plans = torch.zeros((1, 2, 33, 15))
    mask = torch.zeros_like(plans, dtype=torch.bool)
    mask[:, :, 0, 0] = True
    plans[0, 1, 0, 0] = 2
    lateral = torch.tensor([[10.0, 0.0]])
    loss, info = expert_trajectory_loss(prediction, plans, mask, lateral,
                                        torch.ones((1, 2), dtype=torch.bool), lateral_scale=1)
    assert info["chosen_candidate_indices"].tolist() == [1]
    assert loss.item() == pytest.approx(0.75)
    loss.backward()
    assert torch.nonzero(prediction.grad).tolist() == [[0, 917]]


def test_dense_plan_straight_derivatives_and_visibility_mask():
    time = np.arange(301) * 0.05
    pos = np.column_stack((3 * time + 0.2 * time**2, np.zeros_like(time), np.zeros_like(time)))
    vel = np.column_stack((3 + 0.4 * time, np.zeros_like(time), np.zeros_like(time)))
    support = np.ones((1, 33), bool)
    support[:, 20:] = False
    plan, mask = dense_to_plan(time, pos, vel, np.zeros_like(pos), np.array([100]), support)
    np.testing.assert_allclose(plan[0, :, 0], 5 * PLAN_TIMES + 0.2 * PLAN_TIMES**2, atol=1e-5)
    np.testing.assert_allclose(plan[0, :, 3], 5 + 0.4 * PLAN_TIMES, atol=1e-6)
    np.testing.assert_allclose(plan[0, :, 6], 0.4, atol=1e-6)
    assert not mask[0, 20:].any()
    assert mask[0, :20].all()


@pytest.mark.parametrize("sign", [-1, 1])
def test_circle_signs_rotation_unwrap_and_derivatives(sign):
    time = np.arange(301) * 0.05
    omega, radius = sign * 0.35, 8.0
    angle = omega * time
    pos = np.column_stack((radius * np.sin(angle) * sign,
                           radius * (1 - np.cos(angle)) * sign, np.zeros_like(time)))
    vel = np.column_stack((radius * np.cos(angle) * abs(omega),
                           radius * np.sin(angle) * abs(omega), np.zeros_like(time)))
    rpy = np.column_stack((np.zeros_like(time), np.zeros_like(time), np.angle(np.exp(1j * angle))))
    plan, mask = dense_to_plan(time, pos, vel, rpy, np.array([100]), np.ones((1, 33), bool))
    assert mask.all()
    np.testing.assert_allclose(plan[0, :, 11], omega * PLAN_TIMES, atol=1e-6)
    np.testing.assert_allclose(plan[0, :, 14], omega, atol=1e-5)
    assert np.sign(plan[0, 10, 1]) == sign
    np.testing.assert_allclose(plan[0, 0, 3:6], [2.8, 0, 0], atol=1e-6)


def test_banked_current_frame_relative_rotation():
    time = np.arange(301) * 0.05
    from scipy.spatial.transform import Rotation
    rpy = np.tile([0.12, 0.07, 0.9], (len(time), 1))
    forward_frd = Rotation.from_euler("xyz", rpy[0]).apply([4.0, 0, 0])
    world_velocity = forward_frd * [1, 1, -1]
    vel = np.tile(world_velocity, (len(time), 1))
    plan, mask = dense_to_plan(time, time[:, None] * world_velocity, vel, rpy,
                               np.array([100]), np.ones((1, 33), bool))
    np.testing.assert_allclose(plan[0, :, 0], 4 * PLAN_TIMES, atol=1e-5)
    np.testing.assert_allclose(plan[0, :, 1:3], 0, atol=1e-6)
    np.testing.assert_allclose(plan[0, :, 9:15], 0, atol=1e-6)


def test_stock_warps_and_plane_order():
    processor = StockPreprocessor()
    narrow, wide = processor.warps
    assert narrow[0, 0] == pytest.approx(2648 / 910)
    assert wide[0, 0] == pytest.approx(567 / 455)
    assert narrow[1, 2] == pytest.approx(604 - (2648 / 910) * 47.6)
    assert wide[1, 2] == pytest.approx(604 - (567 / 455) * 151.8)
    y = np.arange(16, dtype=np.uint8).reshape(4, 4)
    packed = pack_yuv420(y, np.full((2, 2), 99, np.uint8), np.full((2, 2), 199, np.uint8))
    np.testing.assert_array_equal(packed[:4], np.stack([y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2]]))
    assert packed[4, 0, 0] == 99 and packed[5, 0, 0] == 199


def test_pinned_rgb_producer_range_and_chroma_average():
    rgb = np.array([[[0, 0, 0], [255, 255, 255]],
                    [[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    nv12, layout = image_to_nv12(rgb, "RGB")
    y, u, v = split_nv12(nv12, layout)
    np.testing.assert_array_equal(y, [[16, 235], [81, 41]])
    assert u.item() == 147 and v.item() == 151
    np.testing.assert_array_equal(nv12, image_to_nv12(rgb[..., ::-1], "BGR")[0])
