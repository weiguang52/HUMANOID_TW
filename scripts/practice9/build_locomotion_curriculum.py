#!/usr/bin/env python3
'''Build nested motion manifests without copying motion arrays.'''

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

CATEGORY_RATIOS = {
    'forward': 0.45,
    'turning': 0.25,
    'backward': 0.12,
    'sideways': 0.10,
    'in_place': 0.08,
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise TypeError(f'expected JSON object: {path}')
    return value


def motion_difficulty(motion: dict[str, Any]) -> float:
    quality = motion.get('retarget_quality') or {}
    velocity = quality.get('velocity_ratio_p99_by_joint') or {}
    near = quality.get('near_limit_fraction_by_joint') or {}
    velocity_p99 = max((float(x) for x in velocity.values()), default=0.0)
    near_fraction = max((float(x) for x in near.values()), default=0.0)
    root_xy = float(quality.get('root_xy_speed_p99', 0.0)) / 0.6
    root_yaw = float(quality.get('root_yaw_rate_p99', 0.0)) / 2.0
    body_linear = float(motion.get('body_linear_velocity_error_p95', 0.0)) / 0.05
    body_angular = float(motion.get('body_angular_velocity_error_p95', 0.0)) / 0.2
    return velocity_p99 + root_xy + root_yaw + body_linear + body_angular + 10.0 * near_fraction


def category_targets(count: int) -> dict[str, int]:
    raw = {tag: count * ratio for tag, ratio in CATEGORY_RATIOS.items()}
    result = {tag: math.floor(value) for tag, value in raw.items()}
    remainder = count - sum(result.values())
    order = sorted(raw, key=lambda tag: (-(raw[tag] - result[tag]), tag))
    for tag in order[:remainder]:
        result[tag] += 1
    return result


def validate_motion(record: dict[str, Any], source: Path) -> None:
    motion_id = record.get('id')
    if not isinstance(motion_id, str) or not motion_id:
        raise ValueError(f'invalid motion id in {source}')
    if not record.get('quality_pass') or not record.get('fk_quality_pass'):
        raise ValueError(f'motion {motion_id} did not pass strict quality gates')
    motion_file = Path(str(record.get('file', '')))
    if not motion_file.is_file():
        raise FileNotFoundError(f'motion file is missing for {motion_id}: {motion_file}')


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def build_curriculum(
    full_manifest: dict[str, Any],
    seed_manifest: dict[str, Any],
    selection_manifest: dict[str, Any],
    sizes: list[int],
) -> list[dict[str, Any]]:
    full = {record['id']: dict(record) for record in full_manifest['motions']}
    seed = {record['id']: dict(record) for record in seed_manifest['motions']}
    overlap = set(full) & set(seed)
    if overlap:
        raise ValueError(f'full and seed manifests overlap: {sorted(overlap)}')
    categories = {record['motion_id']: record['tag'] for record in selection_manifest['motions']}
    missing_categories = sorted(set(full) - set(categories))
    if missing_categories:
        raise ValueError(f'missing selection categories for motions: {missing_categories[:5]}')

    combined_count = len(seed) + len(full)
    if sizes != sorted(set(sizes)) or not sizes or sizes[0] < len(seed) or sizes[-1] > combined_count:
        raise ValueError('sizes must be unique, increasing, include all seeds, and fit the combined dataset')

    scores = {motion_id: motion_difficulty(record) for motion_id, record in full.items()}
    pools = {
        tag: sorted(
            (motion_id for motion_id in full if categories[motion_id] == tag),
            key=lambda motion_id: (scores[motion_id], motion_id),
        )
        for tag in CATEGORY_RATIOS
    }
    global_order = sorted(full, key=lambda motion_id: (scores[motion_id], motion_id))
    chosen_full: set[str] = set()
    stages: list[dict[str, Any]] = []

    for size in sizes:
        requested_full = size - len(seed)
        targets = category_targets(requested_full)
        for tag, target in targets.items():
            present = sum(categories[motion_id] == tag for motion_id in chosen_full)
            for motion_id in pools[tag]:
                if present >= target:
                    break
                if motion_id not in chosen_full:
                    chosen_full.add(motion_id)
                    present += 1
        for motion_id in global_order:
            if len(chosen_full) >= requested_full:
                break
            chosen_full.add(motion_id)
        if len(chosen_full) != requested_full:
            raise RuntimeError(f'could not fill curriculum stage {size}')

        records: list[dict[str, Any]] = []
        for motion_id, record in {**seed, **{key: full[key] for key in chosen_full}}.items():
            copied = dict(record)
            copied['curriculum_difficulty'] = scores.get(motion_id, 0.0)
            copied['curriculum_tag'] = categories.get(motion_id, 'seed')
            records.append(copied)
        records.sort(key=lambda record: record['id'])
        stages.append(
            {
                'size': size,
                'motions': records,
                'tag_counts': dict(sorted(Counter(record['curriculum_tag'] for record in records).items())),
                'frames': sum(int(record['frames']) for record in records),
            }
        )
    return stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-manifest', type=Path, required=True)
    parser.add_argument('--seed-manifest', type=Path, required=True)
    parser.add_argument('--selection-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[40, 80, 165, 174])
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.output_dir}')
    full_manifest = read_json(args.full_manifest)
    seed_manifest = read_json(args.seed_manifest)
    selection_manifest = read_json(args.selection_manifest)
    for source, manifest in ((args.full_manifest, full_manifest), (args.seed_manifest, seed_manifest)):
        for record in manifest.get('motions', []):
            validate_motion(record, source)

    stages = build_curriculum(full_manifest, seed_manifest, selection_manifest, args.sizes)
    index: dict[str, Any] = {
        'schema_version': 1,
        'curriculum': 'practice9_balanced_failure_rate_v1',
        'full_manifest': str(args.full_manifest.resolve()),
        'seed_manifest': str(args.seed_manifest.resolve()),
        'selection_manifest': str(args.selection_manifest.resolve()),
        'stages': [],
    }
    root_fields = ('schema_version', 'target_fps', 'robot', 'joint_names')
    for stage in stages:
        manifest = {key: full_manifest[key] for key in root_fields if key in full_manifest}
        manifest['motions'] = stage['motions']
        manifest['failures'] = []
        manifest['curriculum'] = {
            'name': index['curriculum'],
            'stage_size': stage['size'],
            'frames': stage['frames'],
            'tag_counts': stage['tag_counts'],
            'includes_seed_count': len(seed_manifest['motions']),
            'difficulty': 'retarget_velocity_root_fk_near_limit_v1',
        }
        stage_size = stage['size']
        filename = f'stage_{stage_size:03d}.json'
        output = args.output_dir / filename
        atomic_json(output, manifest)
        index['stages'].append(
            {'size': stage_size, 'file': str(output.resolve()), 'frames': stage['frames'], 'tags': stage['tag_counts']}
        )
    atomic_json(args.output_dir / 'curriculum.json', index)
    print(json.dumps(index['stages'], indent=2))


if __name__ == '__main__':
    main()
