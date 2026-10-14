from types import SimpleNamespace
import numpy as np
from lane_dataset.capture import observed_reference
from lane_dataset.geometry import LaneReference, EvidenceFrame


def test_capture_budget_uses_actual_warmup_and_keeps_preview():
    import pytest
    from lane_dataset.capture import recording_budget
    task=dict(warmup_distance_budget_m=30.,planned_preview_m=25.)
    args=dict(warmup_progress_m=22.,recording_start_m=22.,route_end_m=160.,
              ticks=800,maximum_speed_mps=2.55)
    result=recording_budget(task,**args)
    assert result["passed"] and result["required_remaining_route_m"]==pytest.approx(127.)
    with pytest.raises(ValueError,match="initialization exceeded"):
        recording_budget(task,**{**args,"warmup_progress_m":31.})
    with pytest.raises(ValueError,match="remaining route"):
        recording_budget(task,**{**args,"route_end_m":90.})
    assert recording_budget(None,**args) is None


def test_visible_dashed_knots_do_not_borrow_hidden_geometry():
    x=np.arange(30,dtype=float);center=np.column_stack([x,np.zeros_like(x),np.zeros_like(x)])
    left=center.copy();right=center.copy();left[:,1]-=1.75;right[:,1]+=1.75
    reference=LaneReference(left,right)
    visible=np.zeros(30,bool);visible[[8,9,12,13,16,17,20,21]]=True
    refs=tuple((EvidenceFrame(10,.5,"wide"),) if value else () for value in np.tile(visible,2))
    support=SimpleNamespace(supported=np.tile(visible,2),evidence_frames=refs)
    before=observed_reference(reference,support,[0,0,0],.5)
    hidden=~visible;left[hidden,1]+=.4;right[hidden,1]+=.4
    after=observed_reference(LaneReference(left,right),support,[0,0,0],.5)
    np.testing.assert_array_equal(before.center_xyz,after.center_xyz)
    np.testing.assert_array_equal(before.s,after.s)
    q=np.linspace(before.s[0],before.s[-1],50)
    for a,b in zip(before.sample(q),after.sample(q)):
        np.testing.assert_array_equal(a,b)


def test_world_owner_remains_held_through_finalization(monkeypatch,tmp_path):
    import pytest
    from lane_dataset import capture
    from lane_dataset.simulator import world_owner
    def fake_owned(*args):
        with pytest.raises(OSError):
            with world_owner(2400,tmp_path):
                pass
        return 'finalized'
    monkeypatch.setattr(capture,'_capture_owned',fake_owned)
    assert capture.capture_development(tmp_path,2400)=='finalized'
    with world_owner(2400,tmp_path):
        pass


def test_world_owner_releases_after_failure(monkeypatch,tmp_path):
    import pytest
    from lane_dataset import capture
    from lane_dataset.simulator import world_owner
    def fail(*args):raise RuntimeError('capture fixture failed')
    monkeypatch.setattr(capture,'_capture_owned',fail)
    with pytest.raises(RuntimeError,match='fixture'):
        capture.capture_development(tmp_path,2400)
    with world_owner(2400,tmp_path):
        pass


def test_value_error_fallback_cannot_publish_the_unsent_expert_command():
    """Execute the actual capture error branch without starting a simulator."""
    import ast
    import inspect
    from lane_dataset.capture import _capture_owned
    tree=ast.parse(inspect.getsource(_capture_owned))
    handler=next(node for node in ast.walk(tree) if isinstance(node,ast.ExceptHandler)
                 and isinstance(node.type,ast.Name) and node.type.id=="ValueError")
    sent=[]
    from lane_dataset.terminal_suffix import suffix_stop_control
    context=dict(command=SimpleNamespace(accepted=True),observable=object(),suffix_stop_control=suffix_stop_control,
        previous_throttle=.7,previous_brake=0.,longitudinal_reference=8.,integral=1.,
        row={"speed":2.5,"steer":.15,"brake":0.},reason=[],error=ValueError("lost observation"),
        batch_control=lambda *a,**kw:sent.append(kw),client=object(),ego=object())
    exec(compile(ast.fix_missing_locations(ast.Module(body=handler.body,type_ignores=[])),
                 "capture-error-branch","exec"),context)
    assert context["command"] is None and context["observable"] is None
    assert context["previous_throttle"]==0. and context["previous_brake"]==.05
    assert context["longitudinal_reference"]==2.5 and context["integral"]==0.
    assert context["reason"]==["lost observation"] and sent==[{"throttle":0.,"brake":.05,"steer":.15}]
