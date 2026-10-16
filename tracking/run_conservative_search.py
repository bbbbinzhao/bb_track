"""Bounded four-GPU coverage, matched learning and full-validation coordinator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracking.search_policy_experiment import write_json_atomic


def checked(command, log, root):
    print('START ' + str(log), flush=True)
    with log.open('a') as handle:
        handle.write(json.dumps(dict(command=command)) + '\n')
        handle.flush()
        subprocess.run([sys.executable, '-u'] + command, cwd=ROOT, stdout=handle,
                       stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    print('DONE ' + str(log), flush=True)


def mixed_jobs(root, reports):
    jobs = []
    for start in range(0, 200, 5):
        end = min(start + 5, 200)
        directory = root / 'on_policy' / ('shard_%03d_%03d' % (start, end))
        manifest = directory / 'manifest.json'
        if not manifest.exists() or not json.loads(manifest.read_text()).get('complete'):
            jobs.append(('on_policy_%03d' % start,
                ['tracking/search_conservative_coverage.py', 'collect', '--output', str(directory),
                 '--sequence-start', str(start), '--sequence-end', str(end),
                 '--policy-checkpoint', reports['event']['checkpoint'], '--resume']))
    for arm in ('uniform', 'event'):
        data_root = reports[arm]['identity']['cache_identity']['data_root']
        for start in range(0, 90, 6):
            end = min(start + 6, 90)
            output = root / arm / 'val' / ('shard_%03d_%03d' % (start, end))
            manifest = output / 'evaluate_val.json'
            if not manifest.exists() or not json.loads(manifest.read_text()).get('complete'):
                jobs.append((arm + '_val_%03d' % start,
                    ['tracking/search_policy_experiment.py', 'evaluate', '--split', 'val',
                     '--data-root', data_root, '--policy-checkpoint', reports[arm]['checkpoint'],
                     '--sequence-start', str(start), '--sequence-end', str(end),
                     '--output', str(output), '--resume']))
    return jobs


def queue(jobs, gpus, slots, logs):
    workers = []
    try:
        while jobs or workers:
            for item in list(workers):
                process, handle, gpu, tag = item
                if process.poll() is not None:
                    handle.close()
                    workers.remove(item)
                    if process.returncode:
                        raise RuntimeError('Pilot worker failed: ' + tag)
                    print('finished ' + tag, flush=True)
            for gpu in gpus:
                while jobs and sum(w[2] == gpu for w in workers) < slots:
                    tag, command = jobs.pop(0)
                    handle = (logs / (tag + '.log')).open('a')
                    handle.write(json.dumps(dict(command=command, gpu=gpu)) + '\n')
                    handle.flush()
                    process = subprocess.Popen([sys.executable, '-u'] + command, cwd=ROOT,
                        stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ,
                        CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    workers.append((process, handle, gpu, tag))
                    print('launch gpu=%d %s' % (gpu, tag), flush=True)
            time.sleep(5)
    finally:
        for process, handle, _, _ in workers:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--wait-for-coverage', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    logs = root / 'pipeline_logs'
    logs.mkdir(parents=True, exist_ok=True)
    if args.wait_for_coverage:
        while True:
            files = list((root / 'coverage').glob('shard_*/manifest.json'))
            if len(files) == 40 and all(json.loads(p.read_text()).get('complete') for p in files):
                break
            time.sleep(15)
    else:
        checked(['tracking/search_conservative_coverage.py', 'run', '--output', str(root / 'coverage'),
            '--gpus'] + list(map(str, args.gpus)) + ['--workers-per-gpu', str(args.workers_per_gpu)],
            logs / 'coverage.log', root)
    report_path = root / 'teacher_comparison.json'
    reports = json.loads(report_path.read_text()) if report_path.exists() else {}
    if 'uniform' not in reports or 'event' not in reports:
        checked(['tracking/train_conservative_search.py', '--teachers', str(root / 'coverage'),
            '--output', str(root), '--stage', 'initial'], logs / 'train_initial.log', root)
        reports = json.loads(report_path.read_text())
    queue(mixed_jobs(root, reports), args.gpus, args.workers_per_gpu, logs)
    if 'aggregate' not in reports or 'event_continue' not in reports:
        checked(['tracking/train_conservative_search.py', '--teachers', str(root / 'coverage'),
            '--on-policy-teachers', str(root / 'on_policy'), '--output', str(root), '--stage', 'aggregate'],
            logs / 'train_aggregate.log', root)
    checked(['tracking/validate_fixed8_teacher.py', '--root', str(root), '--objectives',
        'uniform', 'event', 'event_continue', 'aggregate', '--gpus'] + list(map(str, args.gpus)) +
        ['--workers-per-gpu', str(args.workers_per_gpu)], logs / 'validation.log', root)
    reports = json.loads(report_path.read_text())
    checked(['tracking/benchmark_search_policy.py', '--include-fixed8', '--policy-checkpoints'] +
        [r['checkpoint'] for r in reports.values()] + ['--policy-labels'] + list(reports) +
        ['--output', str(root / 'isolated_runtime.json')], logs / 'runtime.log', root)
    checked(['tracking/analyze_conservative_search.py', '--root', str(root)], logs / 'analysis.log', root)
    write_json_atomic(str(root / 'pipeline_complete.json'), dict(complete=True, seed=0,
        validation_sequences_per_arm=90, arms=['uniform', 'event', 'event_continue', 'aggregate'],
        test_used=False))
    print('CONSERVATIVE COVERAGE PILOT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
