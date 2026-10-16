"""Repartition stopped teacher shards while retaining completed sequence files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.search_policy_experiment import write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    originals = [root / ('teacher_%03d_%03d' % (start, start + 25)) / 'manifest.json'
                 for start in range(0, 200, 25)]
    payloads = [json.load(path.open()) for path in originals]
    if not all(payload['complete'] for payload in payloads[:5]):
        raise ValueError('This refinement expects the first 125 sequences to be complete')
    known = {}
    for path, payload in zip(originals, payloads):
        for name, record in payload['sequences'].items():
            file_path = (path.parent / record['file']).resolve()
            if not file_path.is_file():
                raise ValueError('Missing completed teacher file: ' + str(file_path))
            known[name] = dict(record, file=str(file_path), source_manifest=str(path))
    names = payloads[0]['all_training_sequences']
    schedule = [{'start': start, 'end': start + 25,
                 'directory': 'teacher_%03d_%03d' % (start, start + 25)}
                for start in range(0, 125, 25)]
    for start in range(125, 200, 5):
        end = start + 5
        directory = 'teacher_%03d_%03d' % (start, end)
        manifest = {key: value for key, value in payloads[0].items() if key not in (
            'sequences', 'complete', 'elapsed_seconds', 'network_state_sha256',
            'frozen_weights_and_buffers_unchanged')}
        manifest.update(sequence_start=start, sequence_end=end, complete=False,
                        sequences={name: known[name] for name in names[start:end] if name in known})
        write_json_atomic(str(root / directory / 'manifest.json'), manifest)
        schedule.append({'start': start, 'end': end, 'directory': directory})
    write_json_atomic(str(root / 'teacher_schedule.json'), {
        'shards': schedule, 'reused_sequences': len(known),
        'source_manifests': [str(path) for path in originals],
        'note': 'Scheduling only; samples and supervision are unchanged. Stop collectors before refinement.'})
    print('Prepared %d shards, reusing %d completed sequences' % (len(schedule), len(known)), flush=True)


if __name__ == '__main__':
    main()
