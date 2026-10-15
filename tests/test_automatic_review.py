import numpy as np
import pytest

from lane_dataset.automatic_review import review_errors


@pytest.mark.parametrize('counts,passed', [([2,5,5,4,4], True), ([1,5,5,5,4], False), ([2,4,4,4,4], False)])
def test_complete_turn_review_allows_balanced_valid_labels_with_short_phase(counts, passed):
    phases = ('approach','entry','apex','exit','recovery')
    ranges = {name:[20*i,20*(i+1)] for i,name in enumerate(phases)}
    records = [dict(label_index=20*i+j,phase=name) for i,name in enumerate(phases) for j in range(counts[i])]
    telemetry = dict(timestamp=np.arange(101)*.05,route_s=np.arange(101))
    review = dict(review_mode='automated_measured_v1',human_reviewed=False,frame_count=101,
                  label_indices=[r['label_index'] for r in records])
    errors = review_errors(review,dict(maneuver_phase_ranges=ranges),telemetry,records)
    assert (not errors) == passed
