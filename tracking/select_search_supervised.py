"""Select supervised checkpoints using only the train-internal development set."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lib.train.data.search_policy import load_teacher_windows
from lib.test.utils.search_region import allowed_actions
from tracking.search_policy_experiment import write_json_atomic


def teacher_statistics(records, actions):
    scores = np.concatenate([record['action_ious'][record['supervision_mask']] for record in records])
    values = torch.tensor(scores[:, actions])
    probabilities = torch.softmax(values / .1, -1)
    entropy = -(probabilities * probabilities.log()).sum(-1).mean()
    marginal = probabilities.mean(0)
    marginal_ce = -(probabilities * marginal.log()).sum(-1).mean()
    return {'steps': len(values), 'mean_action_iou': values.mean(0).tolist(),
            'mean_teacher_probabilities': marginal.tolist(), 'teacher_entropy': float(entropy),
            'uniform_cross_entropy': math.log(len(actions)),
            'constant_distribution_cross_entropy': float(marginal_ce),
            'oracle_iou': float(values.max(-1).values.mean()),
            'best_constant_iou': float(values.mean(0).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root).resolve()
    schedule = json.load((root / 'teacher_schedule.json').open())['shards']
    train, development, identity = load_teacher_windows([
        str(root / shard['directory'] / 'manifest.json') for shard in schedule])
    selection = {'selection_split': 'sequence-disjoint development inside official train',
                 'official_val_used': False, 'test_used': False,
                 'criterion': 'maximum development selected teacher IoU; earliest exact tie',
                 'cache_identity': identity, 'models': {}}
    for mode in ('scale', 'joint'):
        report = json.load((root / ('mlp_' + mode) / 'continued_train/training_report.json').open())
        history = report['history']
        best = max(history, key=lambda entry: entry['development']['selected_iou'])
        epoch = best['epoch']
        folder = 'train' if epoch == 1 else 'continued_train'
        checkpoint = root / ('mlp_' + mode) / folder / ('policy_supervised_seed0_ep%03d.pth' % epoch)
        if not checkpoint.is_file():
            raise ValueError('Missing selected checkpoint')
        diagnostics = {name: teacher_statistics(records, list(allowed_actions(mode)))
                       for name, records in (('train', train), ('development', development))}
        train_prior = np.asarray(diagnostics['train']['mean_teacher_probabilities'])
        dev_prior = np.asarray(diagnostics['development']['mean_teacher_probabilities'])
        diagnostics['development']['train_prior_cross_entropy'] = float(-(
            dev_prior * np.log(train_prior)).sum())
        entropy = diagnostics['development']['teacher_entropy']
        for entry in history:
            entry['development']['kl_to_teacher'] = entry['development']['loss'] - entropy
        selection['models']['mlp_' + mode] = {
            'checkpoint': str(checkpoint), 'epoch': epoch, 'development': best['development'],
            'history': history, 'teacher_statistics': diagnostics,
            'updates_at_selected_epoch': best['updates'],
            'total_supervised_epochs': report['history'][-1]['epoch']}
    path = root / 'convergence_selection.json'
    write_json_atomic(str(path), selection)
    print(json.dumps({name: {'epoch': record['epoch'], 'checkpoint': record['checkpoint'],
                            'development': record['development'],
                            'teacher_statistics': record['teacher_statistics']['development']}
                      for name, record in selection['models'].items()}, indent=2))
    print('Wrote ' + str(path), flush=True)


if __name__ == '__main__':
    main()
