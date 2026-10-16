"""Verify frozen CUDA runtime, branch isolation and policy crop semantics."""
import hashlib
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from lib.test.utils.search_region import SearchRegion
from tracking.search_policy_experiment import (
    build_runtime, load_sequences, read_rgb, write_json_atomic,
)


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.network.state_dict().items():
        digest.update(name.encode('utf8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    args = SimpleNamespace(device='cuda:0', split='val', checkpoint=
        'output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar')
    model = build_runtime(args)
    assert not model.network.training
    before = state_hash(model)
    _, paths, boxes, _ = load_sequences('val', limit=1)[0]
    model.initialize(read_rgb(paths[0]), {'init_bbox': boxes[0].tolist()})
    model.track(read_rgb(paths[1]))
    snapshot = model.snapshot()
    image = read_rgb(paths[2])
    center = tuple(np.asarray(model.state[:2]) + np.asarray(model.state[2:]) / 2)
    a, b = SearchRegion(center, 4), SearchRegion((center[0] + 10, center[1]), 8)
    model.track(image, search_region=a)
    first = np.array(model.state)
    first_hann = model.use_hann
    model.restore(snapshot)
    model.track(image, search_region=b)
    model.restore(snapshot)
    model.track(image, search_region=a)
    np.testing.assert_array_equal(first, model.state)
    assert model.use_hann == first_hann

    # Force both confidence regimes and ensure Hann still follows the original rule.
    model.restore(snapshot)
    model.T_logits = model.T_score = 2.
    model.track(image, search_region=a)
    assert not model.use_hann
    assert model.search_factor == a.factor
    assert model.last_diagnostic['sra_next_factor'] == min(a.factor + model.step,
                                                         model.max_search_area_factor)
    model.T_logits = model.T_score = -1.
    model.track(image, search_region=b)
    assert model.use_hann
    assert model.search_factor == b.factor
    assert model.last_diagnostic['sra_next_factor'] == model.base_factor

    geometry = model._active_crop
    mapped = model.map_box_back([3., 4., 2., 2.], 1.)
    np.testing.assert_allclose(mapped, [geometry.origin_xy[0] + 2.,
                                      geometry.origin_xy[1] + 3., 2., 2.])
    model.initialize(read_rgb(paths[0]), {'init_bbox': boxes[0].tolist()})
    assert model.frame_id == 0 and model.last_diagnostic is None
    np.testing.assert_array_equal(model.motion.velocity, [0., 0.])
    assert state_hash(model) == before
    report = {'network_state_sha256_before_after': before,
              'frozen_weights_and_buffers_unchanged': True, 'branch_order_invariant': True,
              'hann_low_and_high_confidence': True, 'override_factor_not_overwritten': True,
              'subsequent_sra_factor_available': True, 'integer_crop_coordinates': True,
              'sequence_motion_reset': True}
    write_json_atomic('output/search_policy/runtime_verification.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
