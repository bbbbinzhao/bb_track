"""Schedule the seed-zero feasibility pilot across available local GPUs."""
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


def read_complete(path):
    if not path.exists():
        return False
    with path.open() as handle:
        return json.load(handle).get('complete', False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/search_policy/learning_pilot_seed0')
    parser.add_argument('--attach-existing', action='store_true',
                        help='Observe already running fixed controls and first teacher shard')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    python = sys.executable
    schedule_path = output / 'teacher_schedule.json'
    if schedule_path.exists():
        with schedule_path.open() as handle:
            teacher_schedule = json.load(handle)['shards']
    else:
        teacher_schedule = [{'start': start, 'end': start + 25,
                             'directory': 'teacher_%03d_%03d' % (start, start + 25)}
                            for start in range(0, 200, 25)]
    teacher_paths = [output / shard['directory'] for shard in teacher_schedule]
    manifests = [path / 'manifest.json' for path in teacher_paths]
    fixed = [output / ('fixed_%d' % factor) / 'evaluate_val.json' for factor in (4, 6, 8)]
    settings = {'stage': 'one_epoch_supervised_feasibility', 'seed': 0,
                'epochs': 1, 'episodes_per_epoch': 1024, 'batch_size': 4,
                'windows_per_sequence': 8, 'warmup': 4, 'supervision': 8,
                'train_sequences': 200, 'validation_sequences': 90,
                'train_internal_development_fraction': .1,
                'policy_types': ['mlp_scale', 'mlp_joint'],
                'fixed_factors': [4, 6, 8], 'test_used': False,
                'workers_per_gpu': 2,
                'teacher_schedule': teacher_schedule,
                'purpose': 'screen learnability before full multi-seed/temporal/SLT experiments'}
    write_json_atomic(str(output / 'pilot_settings.json'), settings)
    workers = {}

    def launch(gpu, tag, command):
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = str(gpu) if gpu is not None else ''
        environment['OMP_NUM_THREADS'] = '1'
        environment['MKL_NUM_THREADS'] = '1'
        handle = (logs / (tag + '.log')).open('a')
        handle.write(json.dumps({'gpu': gpu, 'command': command}) + '\n')
        handle.flush()
        process = subprocess.Popen([python, '-u'] + command, cwd=str(ROOT), env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        print('launch gpu=%s %s pid=%d' % (gpu, tag, process.pid), flush=True)
        return process, handle, tag

    def reap():
        for gpu, (process, handle, tag) in list(workers.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                del workers[gpu]
                if code:
                    raise RuntimeError('%s failed (%d), see %s' % (tag, code, logs / (tag + '.log')))
                print('finished ' + tag, flush=True)

    try:
        if not args.attach_existing:
            for gpu, factor in enumerate((4, 6, 8)):
                if not read_complete(fixed[gpu]):
                    workers[(gpu, 0)] = launch(gpu, 'fixed_%d' % factor, [
                        'tracking/search_policy_experiment.py', 'evaluate', '--split', 'val',
                        '--fixed-factor', str(factor), '--output', str(fixed[gpu].parent), '--resume'])
            if not read_complete(manifests[0]):
                workers[(3, 0)] = launch(3, 'teacher_0', [
                    'tracking/search_policy_learning.py', 'cache',
                    '--sequence-start', str(teacher_schedule[0]['start']),
                    '--sequence-end', str(teacher_schedule[0]['end']),
                    '--output', str(teacher_paths[0]), '--resume'])
        while not all(read_complete(path) for path in manifests):
            reap()
            busy_tags = {value[2] for value in workers.values()}
            for gpu in range(4):
                for slot in range(2):
                    key = (gpu, slot)
                    if key in workers:
                        continue
                    if slot == 0 and gpu < 3 and not read_complete(fixed[gpu]):
                        continue
                    if slot == 0 and gpu == 3 and not read_complete(manifests[0]):
                        continue
                    pending = [index for index in range(1, len(manifests))
                               if not read_complete(manifests[index]) and 'teacher_%d' % index not in busy_tags]
                    if pending:
                        index = pending[0]
                        tag = 'teacher_%d' % index
                        workers[key] = launch(gpu, tag, ['tracking/search_policy_learning.py', 'cache',
                            '--sequence-start', str(teacher_schedule[index]['start']),
                            '--sequence-end', str(teacher_schedule[index]['end']),
                            '--output', str(teacher_paths[index]), '--resume'])
                        busy_tags.add(tag)
            print('cache progress %d/%d complete shards; fixed controls %d/3 complete' % (
                sum(read_complete(path) for path in manifests),
                len(manifests),
                sum(read_complete(path) for path in fixed)), flush=True)
            time.sleep(20)
        # Atomic manifests can appear just before process exit.
        while workers:
            reap()
            if workers:
                time.sleep(2)
        for action_mode in ('scale', 'joint'):
            train_dir = output / ('mlp_' + action_mode) / 'train'
            checkpoint = train_dir / 'policy_supervised_seed0_ep001.pth'
            if checkpoint.exists():
                continue
            process, handle, tag = launch(None, 'train_mlp_' + action_mode,
                ['tracking/search_policy_learning.py', 'train', '--device', 'cpu',
                 '--policy-type', 'mlp', '--action-mode', action_mode, '--epochs', '1',
                 '--episodes-per-epoch', '1024', '--batch-size', '4', '--seed', '0',
                 '--output', str(train_dir), '--input-manifests'] + [str(p) for p in manifests])
            code = process.wait()
            handle.close()
            if code:
                raise RuntimeError('Training failed: ' + tag)
        jobs = []
        for action_mode in ('scale', 'joint'):
            for shard, (start, end) in enumerate(((0, 23), (23, 46), (46, 69), (69, 90))):
                directory = output / ('mlp_' + action_mode) / 'epoch001' / ('shard%d' % shard)
                report = directory / 'evaluate_val.json'
                if read_complete(report):
                    continue
                jobs.append(('val_%s_%d' % (action_mode, shard),
                    ['tracking/search_policy_experiment.py', 'evaluate', '--split', 'val',
                     '--policy-checkpoint', str(output / ('mlp_' + action_mode) / 'train' /
                                                'policy_supervised_seed0_ep001.pth'),
                     '--sequence-start', str(start), '--sequence-end', str(end),
                     '--output', str(directory), '--resume']))
        while jobs or workers:
            reap()
            for gpu in range(4):
                for slot in range(2):
                    key = (gpu, slot)
                    if key in workers or not jobs:
                        continue
                    if slot == 0 and gpu < 3 and not read_complete(fixed[gpu]):
                        continue
                    tag, command = jobs.pop(0)
                    workers[key] = launch(gpu, tag, command)
            time.sleep(20)
        for action_mode in ('scale', 'joint'):
            reports = [output / ('mlp_' + action_mode) / 'epoch001' / ('shard%d' % shard) /
                       'evaluate_val.json' for shard in range(4)]
            subprocess.run([python, 'tracking/search_policy_experiment.py', 'merge',
                '--output', str(output / ('mlp_' + action_mode)), '--merged-name', 'val_epoch001',
                '--input-reports'] + [str(path) for path in reports], cwd=str(ROOT), check=True)
        subprocess.run([python, 'tracking/analyze_search_learning.py', '--root', str(output)],
                       cwd=str(ROOT), check=True)
        print('PILOT COMPLETE: ' + str(output / 'analysis' / 'learning_pilot_report.md'), flush=True)
    finally:
        for process, handle, _ in workers.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            handle.close()


if __name__ == '__main__':
    main()
