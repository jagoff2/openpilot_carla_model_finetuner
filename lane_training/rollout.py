"""Trainable student recurrence and independent frozen original-model recurrence."""
from __future__ import annotations

import torch

from lane_dataset.loader import student_rollout
from .adapters import model_centerline_loss
from .targets import episode_corridors


def centerline_backward(episode, student, reference, *, plan_adapter, rollout=student_rollout,
                        device="cpu", auxiliary_dtype=None, config=None, progress=None,
                        complete_sequence=False):
    """Accumulate gradients for one accepted episode, without an optimizer step.

    reference is a separate, frozen original checkpoint. It regenerates its own
    recurrent history from the cameras, and its outputs remain fixed as student
    weights change. Student histories come from the student, with detached BPTT.
    No recorded speed or recorded lateral/longitudinal command enters this loss.
    Call optimizer.step() externally after checking the returned diagnostics.
    plan_adapter explicitly decodes the chosen model's output layout. Supply a
    compatible rollout implementation when its inputs/recurrence differ from
    the existing stock-camera sequence ABI.
    """
    if reference is student or any(p.requires_grad for p in reference.parameters()):
        raise ValueError("supply a separate frozen original model")
    student_storage = {p.untyped_storage().data_ptr() for p in student.parameters()}
    if any(p.untyped_storage().data_ptr() in student_storage for p in reference.parameters()):
        raise ValueError("student and original model must not share parameter storage")
    corridors = episode_corridors(episode)
    label_rows = {c.label_index: row for row, c in enumerate(corridors)}
    reference.eval()
    ticks = 0

    @torch.no_grad()
    def original(inputs):
        nonlocal ticks
        result = reference(inputs)
        ticks += 1
        if progress and ticks % 100 == 0:
            progress("reference", ticks, len(episode))
        return result.detach()

    original_predictions, _ = rollout(episode, original, device=device,
        auxiliary_dtype=auxiliary_dtype, detach_history=True,
        stop_after_last_label=not complete_sequence)
    reference_ticks = ticks
    with torch.no_grad():
        _, availability = model_centerline_loss(original_predictions, original_predictions, corridors,
            adapter=plan_adapter, config=config)
    valid_count = availability["valid_labels"]
    if valid_count == 0:
        raise ValueError("episode has no observed geometry along original-model progress")
    student.zero_grad(set_to_none=True)
    ticks, backwards = 0, 0
    losses, rows = [], []

    def supervised(inputs):
        nonlocal ticks, backwards
        output = student(inputs)
        row = label_rows.get(ticks)
        if row is not None:
            loss, diagnostic = model_centerline_loss(output, original_predictions[row:row+1], [corridors[row]],
                adapter=plan_adapter, config=config)
            if diagnostic["valid_labels"]:
                (loss/valid_count).backward()
                backwards += 1
            rows.append(dict(label_index=ticks, **diagnostic))
            losses.append(float(loss.detach()))
        ticks += 1
        if progress and (ticks % 100 == 0 or ticks == len(episode)):
            progress("student", ticks, len(episode))
        return output.detach()

    predictions, _ = rollout(episode, supervised, device=device,
        auxiliary_dtype=auxiliary_dtype, detach_history=True,
        stop_after_last_label=not complete_sequence)
    return predictions, dict(loss=sum(losses)/valid_count, backward_calls=backwards,
        reference_ticks=reference_ticks, student_ticks=ticks, valid_labels=valid_count,
        reference_has_gradients=any(p.grad is not None for p in reference.parameters()),
        recurrent_history="separate original and student histories, both detached",
        label_diagnostics=rows)
