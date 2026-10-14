"""Measured automatic review for unattended corpus generation.

This is explicitly machine review, never an assertion that a human watched a
clip. Admission still independently replays pixels, labels, body clearance,
physical dynamics, complete phases, calibration and source bindings.
"""
import numpy as np


def review_errors(review, manifest, telemetry, label_records):
    errors=[]
    if review.get('review_mode')!='automated_measured_v1' or review.get('human_reviewed') is not False:
        errors.append('automatic_review_mode_invalid')
    if review.get('frame_count')!=len(telemetry['timestamp']):errors.append('automatic_review_frame_count_mismatch')
    indices=[r['label_index'] for r in label_records]
    if review.get('label_indices')!=indices:errors.append('automatic_review_label_indices_mismatch')
    ranges=manifest.get('maneuver_phase_ranges') or {}
    phases={'approach','entry','apex','exit','recovery'}
    if set(ranges)!=phases:
        return errors+['automatic_review_phase_ranges_missing']
    progress=np.asarray(telemetry['route_s'])
    if progress[0]>ranges['approach'][0] or progress[-1]<ranges['recovery'][1]:
        errors.append('automatic_review_complete_maneuver_not_recorded')
    for phase in phases:
        chosen=[r for r in label_records if r.get('phase')==phase]
        if len(chosen)!=4:errors.append('automatic_review_phase_label_count:'+phase)
        lo,hi=ranges[phase]
        if any(not lo<=progress[r['label_index']]<=hi for r in chosen):
            errors.append('automatic_review_phase_label_location:'+phase)
    return errors
