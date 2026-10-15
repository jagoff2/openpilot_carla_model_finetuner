"""Performance changes must preserve camera bytes and causal pixel evidence."""
import numpy as np
import pytest

from lane_dataset.preprocessing import image_to_nv12, _rgb_producer
from lane_dataset.visibility import EvidenceHistory
from test_visibility import evidence


@pytest.mark.parametrize('shape', [(18, 34), (1208, 1928)])
@pytest.mark.parametrize('order', ['RGB', 'BGR'])
def test_tiled_rgb_conversion_is_byte_identical_to_pinned_producer(shape, order):
    rng = np.random.default_rng(712)
    # Include a negative-stride image, as CARLA camera conversion can use views.
    image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)[:, ::-1]
    expected, layout = _rgb_producer()(image if order == 'RGB' else image[..., ::-1])
    actual, actual_layout = image_to_nv12(image, order)
    assert actual_layout == layout
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


def test_sparse_history_preserves_identity_order_and_source_only_visibility():
    history = EvidenceHistory()
    ids = tuple(f'p{i}' for i in range(2000))
    visible = np.zeros(len(ids), bool)
    visible[[13, 1900]] = True
    for frame in range(3):
        for camera in ('narrow', 'wide'):
            history.admit(evidence(frame, frame*.05, camera, visible, ids=ids))
    current_ids = ('outside_query', 'p1900', 'p13')
    for camera in ('narrow', 'wide'):
        history.admit(evidence(10, .5, camera, [True, False, True],
                               [True, True, True], ids=current_ids))
    result = history.support(10, .5, ('p1900', 'p13'), np.ones(2, bool),
                             np.array([2., 0.]), ['p13'], distance_m=np.array([20., 5.]))
    assert result.category == 'history_crop_loss'
    assert result.current_visible.tolist() == [False, True]
    assert result.current_source_visible.tolist() == [True, True]
    assert result.visibility_duration_s.tolist() == [.1, .1]
    expected = [(i, camera) for i in range(3) for camera in ('narrow', 'wide')]
    assert [(ref.frame_id, ref.camera) for ref in result.evidence_frames[0]] == expected
    assert [(ref.frame_id, ref.camera) for ref in result.evidence_frames[1]] == expected + [(10, 'narrow'), (10, 'wide')]
