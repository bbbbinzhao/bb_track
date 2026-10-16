"""Validate train-development-selected policies alongside the initial pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def complete(path):
    return path.is_file() and json.load(path.open()).get('complete', False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    selection = json.load((root / 'convergence_selection.json').open())
    if selection['official_val_used'] or selection['test_used']:
        raise ValueError('Checkpoint selection must use train-internal development only')
    initial = []
    for mode in ('scale', 'joint'):
        for shard in range(4):
            log = root / 'logs' / ('val_%s_%d.log' % (mode, shard))
            with log.open() as handle:
                header = json.loads(handle.readline())
            initial.append((header['gpu'], root / ('mlp_' + mode) / 'epoch001' /
                            ('shard%d' % shard) / 'evaluate_val.json'))
    jobs = []
    for mode in ('scale', 'joint'):
        for shard, (start, end) in enumerate(((0, 23), (23, 46), (46, 69), (69, 90))):
            directory = root / ('mlp_' + mode) / 'selected_val' / ('shard%d' % shard)
            if not complete(directory / 'evaluate_val.json'):
                jobs.append((mode, shard, start, end, directory))
    workers = []
    try:
        while jobs or workers:
            for worker in list(workers):
                process, handle, gpu, tag = worker
                code = process.poll()
                if code is not None:
                    handle.close()
                    workers.remove(worker)
                    if code:
                        raise RuntimeError('Selected policy validation failed: ' + tag)
                    print('finished ' + tag, flush=True)
            for gpu in range(4):
                busy = sum(device == gpu and not complete(path) for device, path in initial)
                busy += sum(worker[2] == gpu for worker in workers)
                while jobs and busy < 2:
                    mode, shard, start, end, directory = jobs.pop(0)
                    tag = 'selected_%s_%d' % (mode, shard)
                    command = [sys.executable, '-u', 'tracking/search_policy_experiment.py',
                        'evaluate', '--split', 'val', '--policy-checkpoint',
                        selection['models']['mlp_' + mode]['checkpoint'],
                        '--sequence-start', str(start), '--sequence-end', str(end),
                        '--output', str(directory), '--resume']
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    handle = (root / 'logs' / (tag + '.log')).open('a')
                    handle.write(json.dumps({'gpu': gpu, 'command': command}) + '\n')
                    handle.flush()
                    process = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                                               stdout=handle, stderr=subprocess.STDOUT)
                    workers.append((process, handle, gpu, tag))
                    busy += 1
                    print('launch gpu=%d %s' % (gpu, tag), flush=True)
            time.sleep(15)
        for mode in ('scale', 'joint'):
            reports = [root / ('mlp_' + mode) / 'selected_val' / ('shard%d' % shard) /
                       'evaluate_val.json' for shard in range(4)]
            subprocess.run([sys.executable, 'tracking/search_policy_experiment.py', 'merge',
                '--output', str(root / ('mlp_' + mode)), '--merged-name', 'val_selected',
                '--input-reports'] + [str(path) for path in reports], cwd=str(ROOT), check=True)
        # The initial validation is normally complete by this point.
        while not all(complete(path) for _, path in initial):
            time.sleep(15)
        subprocess.run([sys.executable, 'tracking/analyze_search_learning.py', '--root', str(root)],
                       cwd=str(ROOT), check=True)
        print('SELECTED VALIDATION COMPLETE', flush=True)
    finally:
        for process, handle, _, _ in workers:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            handle.close()


if __name__ == '__main__':
    main()
