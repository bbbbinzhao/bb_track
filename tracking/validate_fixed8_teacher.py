"""Complete-val evaluation of the two train-development-selected teacher models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--objectives', nargs='+', default=['one', 'eight'])
    parser.add_argument('--wait-for-root')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    reports = json.loads((root / 'teacher_comparison.json').read_text())
    jobs = []
    manifests = {}
    (root / 'logs').mkdir(exist_ok=True)
    for objective in args.objectives:
        report = reports[objective]
        if report['official_val_used'] or report['test_used']:
            raise ValueError('Checkpoint selection used evaluation labels')
        manifests[objective] = []
        for start in range(0, 90, 6):
            end = min(start + 6, 90)
            output = root / objective / 'val' / ('shard_%03d_%03d' % (start, end))
            path = output / 'evaluate_val.json'
            manifests[objective].append(path)
            if not path.exists() or not json.loads(path.read_text()).get('complete', False):
                jobs.append((objective, start, end, output))
    workers = []
    try:
        while workers or jobs:
            for item in list(workers):
                process, handle, gpu, tag = item
                if process.poll() is not None:
                    handle.close()
                    workers.remove(item)
                    if process.returncode:
                        raise RuntimeError('Validation failed: ' + tag)
                    print('finished ' + tag, flush=True)
            occupied = {}
            if args.wait_for_root:
                dependency = Path(args.wait_for_root)
                logs = [dependency / 'logs' / ('%s_%03d_%03d.log' % (objective, start, min(start + 6, 90)))
                        for objective in ('one', 'eight') for start in range(0, 90, 6)]
                # Wait until the other scheduler has dispatched its full queue, then fill free slots.
                if not all(path.exists() and path.stat().st_size for path in logs):
                    time.sleep(5)
                    continue
                for path in logs:
                    with path.open() as handle:
                        header = json.loads(handle.readline())
                    command = header['command']
                    report_path = Path(command[command.index('--output') + 1]) / 'evaluate_val.json'
                    complete = report_path.exists() and json.loads(report_path.read_text()).get('complete', False)
                    if not complete:
                        occupied[header['gpu']] = occupied.get(header['gpu'], 0) + 1
            for gpu in args.gpus:
                busy = sum(item[2] == gpu for item in workers) + occupied.get(gpu, 0)
                while jobs and busy < args.workers_per_gpu:
                    objective, start, end, output = jobs.pop(0)
                    report = reports[objective]
                    tag = '%s_%03d_%03d' % (objective, start, end)
                    command = [sys.executable, '-u', 'tracking/search_policy_experiment.py', 'evaluate',
                        '--policy-checkpoint', report['checkpoint'], '--split', 'val',
                        '--data-root', report['identity']['cache_identity']['data_root'],
                        '--sequence-start', str(start), '--sequence-end', str(end),
                        '--output', str(output), '--resume']
                    handle = (root / 'logs' / (tag + '.log')).open('a')
                    handle.write(json.dumps(dict(gpu=gpu, command=command)) + '\n')
                    handle.flush()
                    process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ,
                        CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                        stdout=handle, stderr=subprocess.STDOUT)
                    workers.append((process, handle, gpu, tag))
                    busy += 1
                    print('launch gpu=%d %s' % (gpu, tag), flush=True)
            time.sleep(5)
    finally:
        for process, handle, _, _ in workers:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            handle.close()
    for objective, paths in manifests.items():
        subprocess.run([sys.executable, 'tracking/search_policy_experiment.py', 'merge',
            '--output', str(root / objective), '--merged-name', 'val_selected',
            '--input-reports'] + [str(path) for path in paths], cwd=ROOT, check=True)
    print('MATCHED TEACHER VALIDATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
