"""Explicit CD210 backward pass for independently admitted behavior sequences.

No optimizer, production activation, file writes, or recorded control inputs.
The original frozen checkpoint generates its own fixed progress and recurrent
history; student recurrence remains detached between densely replayed frames.
"""
from __future__ import annotations

import torch

from lane_training.cd210 import ROLLOUT_PLAN, student_rollout
from .loader import AdmittedBehaviorSequence
from .loss import model_behavior_loss


def behavior_backward(episode,student,reference,*,plan_adapter=ROLLOUT_PLAN,rollout=student_rollout,
                      device='cpu',auxiliary_dtype=None,config=None,progress=None,complete_sequence=False):
    """Accumulate student gradients using exact admitted label-to-corridor rows.

    Requires a loaded AdmittedBehaviorSequence. The caller may perform an
    optimizer step only after inspecting diagnostics; this function never does.
    Both parameters and buffers must have storage separate from the reference.
    """
    if not isinstance(episode,AdmittedBehaviorSequence):
        raise TypeError('behavior training requires an independently admitted BehaviorSequence')
    episode.assert_admitted()
    if (student is reference or any(p.requires_grad for p in reference.parameters())
            or any(p.grad is not None for p in reference.parameters())):
        raise ValueError('supply a separate frozen original model without gradients')
    def storage(model):
        return {v.untyped_storage().data_ptr() for v in (*model.parameters(),*model.buffers()) if v.numel()}
    if storage(student)&storage(reference):
        raise ValueError('student and original model must not share parameter or buffer storage')
    labels=episode.behavior_labels
    corridors=[label.corridor for label in labels]
    rows={label.label_index:row for row,label in enumerate(labels)}
    if len(rows)!=len(labels):raise ValueError('duplicate behavior label mapping')
    reference.eval();ticks=0

    @torch.no_grad()
    def original(inputs):
        nonlocal ticks
        result=reference(inputs);ticks+=1
        if progress and ticks%100==0:progress('reference',ticks,len(episode))
        return result.detach()

    originals,_=rollout(episode,original,device=device,auxiliary_dtype=auxiliary_dtype,
        detach_history=True,stop_after_last_label=not complete_sequence)
    reference_ticks=ticks
    if originals.shape[0]!=len(labels):
        raise ValueError('original rollout did not preserve exact admitted label rows')
    with torch.no_grad():
        _,availability=model_behavior_loss(originals,originals,corridors,adapter=plan_adapter,config=config)
    valid_count=availability['valid_labels']
    if not valid_count:raise ValueError('no admitted behavior geometry along frozen-original progress')
    episode.assert_admitted()
    student.zero_grad(set_to_none=True)
    ticks=backwards=0;losses=[];diagnostics=[]

    def supervised(inputs):
        nonlocal ticks,backwards
        output=student(inputs);row=rows.get(ticks)
        if row is not None:
            loss,diagnostic=model_behavior_loss(output,originals[row:row+1],[corridors[row]],
                adapter=plan_adapter,config=config)
            if diagnostic['valid_labels']:
                (loss/valid_count).backward();backwards+=1
            losses.append(float(loss.detach()))
            diagnostics.append(dict(label_index=ticks,record_sha256=labels[row].record_sha256,**diagnostic))
        ticks+=1
        if progress and (ticks%100==0 or ticks==len(episode)):progress('student',ticks,len(episode))
        return output.detach()

    try:
        predictions,_=rollout(episode,supervised,device=device,auxiliary_dtype=auxiliary_dtype,
            detach_history=True,stop_after_last_label=not complete_sequence)
        episode.assert_admitted()
        if predictions.shape[0]!=len(labels) or len(diagnostics)!=len(labels):
            raise ValueError('student rollout did not preserve exact admitted label rows')
    except BaseException:
        # A late review hold or damaged frame must not leave partial gradients
        # available to a caller that catches the exception and keeps running.
        student.zero_grad(set_to_none=True)
        raise
    return predictions,dict(loss=sum(losses)/valid_count,backward_calls=backwards,valid_labels=valid_count,
        reference_ticks=reference_ticks,student_ticks=ticks,
        reference_has_gradients=any(p.grad is not None for p in reference.parameters()),
        recurrent_history='independent original/student CD210 histories, detached between frames',
        objective='unmarked_right_half_observed_avoidance_and_recovery',label_diagnostics=diagnostics,
        optimizer_step_performed=False,production_activation_performed=False)
