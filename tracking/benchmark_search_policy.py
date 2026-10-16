"""Matched isolated latency for SRA and optional search controllers."""
import os
import argparse
import hashlib
import statistics
import sys
from types import SimpleNamespace

import cv2
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from lib.models.search_policy import SearchPolicy
from tracking.search_policy_experiment import build_runtime, load_sequences, run_sequence, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-checkpoints', nargs='+')
    parser.add_argument('--policy-labels', nargs='+')
    parser.add_argument('--include-fixed8', action='store_true')
    parser.add_argument('--cpu-core', type=int)
    parser.add_argument('--output', default='output/search_policy/isolated_runtime.json')
    options = parser.parse_args()
    if options.cpu_core is not None:
        if options.cpu_core not in os.sched_getaffinity(0):
            parser.error('Requested CPU core is outside the allowed affinity')
        os.sched_setaffinity(0, {options.cpu_core})
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    args = SimpleNamespace(device='cuda:0', split='val', checkpoint=
        'output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar')
    model = build_runtime(args)
    sequence_name, frames, boxes, visible = load_sequences('val', limit=1)[0]
    frames, boxes, visible = frames[:200], boxes[:200], visible[:200]
    policies, action_modes = {}, {}
    if options.policy_labels and (not options.policy_checkpoints or
            len(options.policy_labels) != len(options.policy_checkpoints)):
        parser.error('One label per policy checkpoint is required')
    if options.policy_checkpoints:
        for index, path in enumerate(options.policy_checkpoints):
            checkpoint = torch.load(path, map_location='cpu')
            with open(args.checkpoint, 'rb') as handle:
                if hashlib.sha256(handle.read()).hexdigest() != checkpoint['backbone_sha256']:
                    raise ValueError('Policy uses a different frozen tracker')
            name = ('gru_' if checkpoint['recurrent'] else 'mlp_') + checkpoint['action_mode']
            if options.policy_labels:
                name = options.policy_labels[index]
            if name in ('sra', 'fixed8'):
                raise ValueError('The sra and fixed8 labels are reserved for baselines')
            if name in policies:
                raise ValueError('Benchmark expects distinct policy types/action modes')
            policy = SearchPolicy(recurrent=checkpoint['recurrent'],
                                  mlp_depth=checkpoint.get('mlp_depth', 3)).to(args.device).eval()
            policy.load_state_dict(checkpoint['policy'])
            policy.decision = checkpoint.get('decision')
            policies[name] = policy
            action_modes[name] = checkpoint['action_mode']
    runs = {'sra': [], **{name: [] for name in policies}} if policies else {'sra': [], 'motion': []}
    if options.include_fixed8:
        runs['fixed8'] = []
    names = list(runs)
    for repeat in range(3):
        order = (names[repeat:] + names[:repeat]) if policies else (
            names if repeat % 2 == 0 else list(reversed(names)))
        for mode in order:
            model.diagnostics_enabled = mode != 'sra'
            torch.cuda.synchronize()
            result = run_sequence(model, frames, boxes, visible, motion_baseline=mode == 'motion',
                                  policy=policies.get(mode), action_mode=action_modes.get(mode, 'joint'),
                                  fixed_factor=8. if mode == 'fixed8' else None)
            torch.cuda.synchronize()
            seconds = result['times'][21:]
            runs[mode].append(float(len(seconds) / seconds.sum()))
            print('%s repeat=%d fps=%.3f' % (mode, repeat + 1, runs[mode][-1]), flush=True)
    summary = {mode: statistics.median(values) for mode, values in runs.items()}
    if not policies:
        summary['motion_fps_change_percent'] = 100 * (summary['motion'] / summary['sra'] - 1)
    payload = {'median_fps': summary, 'runs_fps': runs,
               'frames_per_repeat': 200, 'warmup_frames_excluded': 20,
               'sequence_name': sequence_name, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'repeats': 3, 'gpu': torch.cuda.get_device_name(),
               'cpu_threads': 1, 'timing': 'tracking and controller; image decoding excluded',
               'feature_extraction': 'disabled for deployed SRA; included for controllers',
               'policy_checkpoints': options.policy_checkpoints,
               'policy_labels': options.policy_labels,
               'relative_fps_change_percent': {name: 100 * (summary[name] / summary['sra'] - 1)
                                               for name in runs if name != 'sra'},
               'relative_to_fixed8_percent': {name: 100 * (summary[name] / summary['fixed8'] - 1)
                    for name in policies} if options.include_fixed8 else None,
               'scope': 'isolated runtime benchmark; full-val accuracy is reported separately'}
    write_json_atomic(options.output, payload)
    print(payload, flush=True)


if __name__ == '__main__':
    main()
