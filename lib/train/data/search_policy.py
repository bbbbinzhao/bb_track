"""Consecutive, causal observation windows for frozen-tracker supervision."""
import hashlib
import os

import numpy as np
import torch


CACHE_PROTOCOL = 'consecutive_search_teacher_v1'
WARMUP = 4
SUPERVISION = 8
WINDOW = WARMUP + SUPERVISION


def window_starts(frame_count, windows_per_sequence=8):
    # Observation t chooses the crop for t+1; frame zero has no response map.
    last = frame_count - WINDOW - 1
    if last < 1:
        return []
    return np.unique(np.linspace(1, last, min(windows_per_sequence, last), dtype=int)).tolist()


def development_sequences(sequence_names, fraction=0.1):
    ranked = sorted(sequence_names, key=lambda name: hashlib.sha256(
        ('search-policy-dev-v1:' + name).encode()).hexdigest())
    count = max(1, int(round(len(ranked) * fraction)))
    return set(ranked[:count])


def load_teacher_windows(manifest_paths):
    import json

    records, metadata, seen = [], None, set()
    for path in manifest_paths:
        with open(path) as handle:
            manifest = json.load(handle)
        if manifest.get('protocol') != CACHE_PROTOCOL or not manifest.get('complete'):
            raise ValueError('Expected a complete consecutive teacher cache: ' + path)
        if manifest.get('split') != 'train':
            raise ValueError('Validation/test labels cannot train a search policy')
        identity = {key: manifest[key] for key in (
            'protocol', 'split', 'data_root', 'backbone_sha256', 'annotations_sha256',
            'windows_per_sequence', 'warmup', 'supervision', 'teacher_temperature',
            'all_training_sequences')}
        if metadata is not None and identity != metadata:
            raise ValueError('Teacher shard identities differ')
        metadata = identity
        for name, value in manifest['sequences'].items():
            if name in seen:
                raise ValueError('Duplicate global sequence name: ' + name)
            seen.add(name)
            chunk = torch.load(os.path.join(os.path.dirname(path), value['file']), map_location='cpu')
            if chunk['sequence_name'] != name or chunk['protocol'] != CACHE_PROTOCOL:
                raise ValueError('Teacher sequence identity mismatch')
            n = len(chunk['observations'])
            for index in range(n):
                ids = np.asarray(chunk['frame_ids'][index])
                if len(ids) != WINDOW or not np.all(np.diff(ids) == 1):
                    raise ValueError('Teacher windows must contain consecutive frames')
                if not np.isfinite(chunk['observations'][index]).all():
                    raise ValueError('Nonfinite policy observation')
                record = {key: np.asarray(chunk[key][index]) for key in (
                    'observations', 'action_ious', 'supervision_mask', 'frame_ids')}
                record['sequence_name'] = name
                records.append(record)
    if not records:
        raise ValueError('No continuous teacher windows found')
    if seen != set(metadata['all_training_sequences']):
        raise ValueError('Teacher manifests do not cover the complete training split')
    dev = development_sequences(sorted(seen))
    train = [record for record in records if record['sequence_name'] not in dev]
    validation = [record for record in records if record['sequence_name'] in dev]
    return train, validation, dict(metadata, development_sequences=sorted(dev),
                                   cached_sequences=sorted(seen))


def collate_windows(records, device):
    return {key: torch.as_tensor(np.stack([record[key] for record in records]), device=device)
            for key in ('observations', 'action_ious', 'supervision_mask')}


def supervised_policy_loss(policy, batch, action_ids, temperature=0.1):
    hidden, logits = None, []
    for step in range(WINDOW):
        step_logits, hidden = policy(batch['observations'][:, step].float(), hidden)
        logits.append(step_logits.index_select(-1, action_ids))
    logits = torch.stack(logits, dim=1)
    scores = batch['action_ious'].float().index_select(-1, action_ids)
    teacher = torch.softmax(scores / temperature, dim=-1)
    mask = batch['supervision_mask'].bool().clone()
    mask[:, :WARMUP] = False
    if not mask.any():
        return None, logits, scores, mask
    losses = -(teacher * torch.log_softmax(logits, dim=-1)).sum(-1)
    return losses[mask].mean(), logits, scores, mask
