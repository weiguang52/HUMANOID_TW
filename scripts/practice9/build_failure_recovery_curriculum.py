#!/usr/bin/env python3
'''Build a derived motion manifest from terminal failure traces.

Early failures caused by an unsupported airborne prefix are logically trimmed
with ``start_frame``. Later repeatable failures receive a local sampling prior
through ``sampling_intervals``. Source NPZ files are never modified or copied.
'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise TypeError(f'Expected a JSON object: {path}')
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_motion_path(manifest_path: Path, item: dict[str, Any]) -> Path:
    path = Path(str(item['file'])).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def first_true_run(mask: np.ndarray, window: int) -> int | None:
    if mask.ndim != 1 or window <= 0:
        raise ValueError('mask must be one-dimensional and window must be positive')
    run = 0
    for index, value in enumerate(mask.tolist()):
        run = run + 1 if value else 0
        if run >= window:
            return index - window + 1
    return None


def find_stable_support_start(
    path: Path,
    *,
    current_start: int,
    current_end: int | None,
    support_height: float,
    support_speed: float,
    support_window_frames: int,
) -> tuple[int, int, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = {'joint_pos', 'body_names', 'body_pos_w', 'body_lin_vel_w'}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f'{path}: missing support-detection fields {missing}')
        source_length = int(np.asarray(data['joint_pos']).shape[0])
        end = source_length if current_end is None else current_end
        if current_start < 0 or end > source_length or end - current_start < 2:
            raise ValueError(f'{path}: invalid existing slice [{current_start}:{end}]')
        names = [str(value) for value in np.asarray(data['body_names']).tolist()]
        try:
            foot_ids = [names.index('left_foot'), names.index('right_foot')]
        except ValueError as exc:
            raise ValueError(f'{path}: left_foot/right_foot are required') from exc
        positions = np.asarray(data['body_pos_w'], dtype=np.float32)[current_start:end, foot_ids]
        velocities = np.asarray(data['body_lin_vel_w'], dtype=np.float32)[current_start:end, foot_ids]
    if positions.shape != velocities.shape or positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f'{path}: invalid foot position/velocity shapes')
    speed = np.linalg.norm(velocities, axis=2)
    supported = ((positions[:, :, 2] <= support_height) & (speed <= support_speed)).any(axis=1)
    local_start = first_true_run(supported, support_window_frames)
    if local_start is None:
        raise ValueError(f'{path}: no stable support window found')
    absolute_start = current_start + local_start
    if end - absolute_start < 2:
        raise ValueError(f'{path}: stable support leaves fewer than two frames')
    foot_z = positions[local_start, :, 2]
    foot_speed = speed[local_start]
    details = {
        'support_height_m': support_height,
        'support_speed_mps': support_speed,
        'support_window_frames': support_window_frames,
        'support_foot_z_m': [float(value) for value in foot_z],
        'support_foot_speed_mps': [float(value) for value in foot_speed],
    }
    return absolute_start, end, details


def _validate_parameters(args: argparse.Namespace) -> None:
    positive = {
        'support_height': args.support_height,
        'support_speed': args.support_speed,
        'phase_multiplier': args.phase_multiplier,
    }
    if not all(math.isfinite(value) and value > 0.0 for value in positive.values()):
        raise ValueError(f'Positive finite parameters required: {positive}')
    integer_values = {
        'support_window_frames': args.support_window_frames,
        'early_failure_max_frame': args.early_failure_max_frame,
        'phase_pre_frames': args.phase_pre_frames,
        'phase_post_frames': args.phase_post_frames,
    }
    if any(value < 0 for value in integer_values.values()) or args.support_window_frames == 0:
        raise ValueError(f'Invalid frame parameters: {integer_values}')


def build_recovery_manifest(
    source: dict[str, Any],
    traces: dict[str, Any],
    *,
    source_path: Path,
    traces_path: Path,
    support_height: float,
    support_speed: float,
    support_window_frames: int,
    early_failure_max_frame: int,
    phase_pre_frames: int,
    phase_post_frames: int,
    phase_multiplier: float,
) -> dict[str, Any]:
    motions = source.get('motions')
    records = traces.get('records')
    if source.get('schema_version') != 1 or not isinstance(motions, list) or not motions:
        raise ValueError('Source must be a non-empty schema-v1 motion manifest')
    if traces.get('schema_version') != 1 or not isinstance(records, list) or not records:
        raise ValueError('Failure traces must be a non-empty schema-v1 report')
    if traces.get('truncated') is not False or int(traces.get('records_captured', -1)) != len(records):
        raise ValueError('Failure traces must be complete and untruncated')

    source_sha = sha256(source_path)
    accepted_trace_manifest_shas = {source_sha}
    curriculum = source.get('curriculum')
    if isinstance(curriculum, dict) and isinstance(curriculum.get('source_manifest_sha256'), str):
        accepted_trace_manifest_shas.add(curriculum['source_manifest_sha256'])
    if traces.get('manifest_sha256') not in accepted_trace_manifest_shas:
        raise ValueError('Failure traces are not bound to the source manifest lineage')

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise TypeError('Failure trace record must be an object')
        motion_index = record.get('motion_index')
        if isinstance(motion_index, bool) or not isinstance(motion_index, int):
            raise TypeError(f'Invalid trace motion_index: {motion_index}')
        if motion_index < 0 or motion_index >= len(motions):
            raise ValueError(f'Trace motion_index out of range: {motion_index}')
        motion_id = str(motions[motion_index].get('id', ''))
        if record.get('motion_id') != motion_id:
            raise ValueError(f'Trace record does not match source motion {motion_index}')
        terminal = record.get('terminal')
        frame = terminal.get('motion_frame') if isinstance(terminal, dict) else None
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError(f'Invalid terminal frame for {motion_id}: {frame}')
        grouped[motion_index].append(record)

    copied_motions: list[dict[str, Any]] = []
    trimmed: list[str] = []
    phase_weighted: list[str] = []
    for motion_index, motion in enumerate(motions):
        if not isinstance(motion, dict):
            raise TypeError(f'Invalid source motion entry {motion_index}')
        copied = dict(motion)
        motion_id = str(copied.get('id', ''))
        motion_records = grouped.get(motion_index)
        if not motion_records:
            copied_motions.append(copied)
            continue
        if copied.get('sampling_intervals') is not None:
            raise ValueError(f'{motion_id}: refusing to overwrite existing sampling_intervals')
        terminal_frames = sorted(int(record['terminal']['motion_frame']) for record in motion_records)
        length = int(copied['frames'])
        if terminal_frames[-1] >= length:
            raise ValueError(f'{motion_id}: terminal frame exceeds motion length')

        if terminal_frames[-1] <= early_failure_max_frame:
            current_start = int(copied.get('start_frame', 0))
            current_end_value = copied.get('end_frame')
            current_end = None if current_end_value is None else int(current_end_value)
            absolute_start, absolute_end, support = find_stable_support_start(
                resolve_motion_path(source_path, copied),
                current_start=current_start,
                current_end=current_end,
                support_height=support_height,
                support_speed=support_speed,
                support_window_frames=support_window_frames,
            )
            local_trim = absolute_start - current_start
            if local_trim <= terminal_frames[-1]:
                raise ValueError(f'{motion_id}: support trim does not remove the early failure region')
            copied['start_frame'] = absolute_start
            if current_end is not None:
                copied['end_frame'] = current_end
            copied['frames'] = absolute_end - absolute_start
            copied['recovery'] = {
                'mode': 'trim_unstable_prefix',
                'terminal_frames': terminal_frames,
                'trimmed_source_frames': local_trim,
                **support,
            }
            trimmed.append(motion_id)
        else:
            interval_start = max(0, terminal_frames[0] - phase_pre_frames)
            interval_end = min(length - 1, terminal_frames[-1] + phase_post_frames + 1)
            if interval_end <= interval_start:
                raise ValueError(f'{motion_id}: empty failure-phase sampling interval')
            copied['sampling_intervals'] = [
                {
                    'start': interval_start,
                    'end': interval_end,
                    'multiplier': phase_multiplier,
                    'reason': 'repeatable_terminal_failure_phase',
                }
            ]
            copied['recovery'] = {
                'mode': 'phase_sampling_prior',
                'terminal_frames': terminal_frames,
                'phase_pre_frames': phase_pre_frames,
                'phase_post_frames': phase_post_frames,
                'phase_multiplier': phase_multiplier,
            }
            phase_weighted.append(motion_id)
        copied_motions.append(copied)

    result = {key: value for key, value in source.items() if key not in {'motions', 'curriculum'}}
    result['motions'] = copied_motions
    result['curriculum'] = {
        'name': 'practice9_failure_recovery_v1',
        'source_manifest': str(source_path.resolve()),
        'source_manifest_sha256': source_sha,
        'failure_traces': str(traces_path.resolve()),
        'failure_traces_sha256': sha256(traces_path),
        'checkpoint': traces.get('checkpoint'),
        'checkpoint_sha256': traces.get('checkpoint_sha256'),
        'trimmed_motion_ids': trimmed,
        'phase_weighted_motion_ids': phase_weighted,
        'parameters': {
            'support_height_m': support_height,
            'support_speed_mps': support_speed,
            'support_window_frames': support_window_frames,
            'early_failure_max_frame': early_failure_max_frame,
            'phase_pre_frames': phase_pre_frames,
            'phase_post_frames': phase_post_frames,
            'phase_multiplier': phase_multiplier,
        },
        'motion_count': len(copied_motions),
        'frames': sum(int(motion['frames']) for motion in copied_motions),
    }
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to overwrite output: {path}')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--failure-traces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--support-height', type=float, default=0.05)
    parser.add_argument('--support-speed', type=float, default=0.15)
    parser.add_argument('--support-window-frames', type=int, default=15)
    parser.add_argument('--early-failure-max-frame', type=int, default=50)
    parser.add_argument('--phase-pre-frames', type=int, default=100)
    parser.add_argument('--phase-post-frames', type=int, default=50)
    parser.add_argument('--phase-multiplier', type=float, default=4.0)
    args = parser.parse_args()
    _validate_parameters(args)

    source_path = args.source_manifest.expanduser().resolve()
    traces_path = args.failure_traces.expanduser().resolve()
    result = build_recovery_manifest(
        read_json(source_path),
        read_json(traces_path),
        source_path=source_path,
        traces_path=traces_path,
        support_height=args.support_height,
        support_speed=args.support_speed,
        support_window_frames=args.support_window_frames,
        early_failure_max_frame=args.early_failure_max_frame,
        phase_pre_frames=args.phase_pre_frames,
        phase_post_frames=args.phase_post_frames,
        phase_multiplier=args.phase_multiplier,
    )
    atomic_json(args.output, result)
    print(json.dumps(result['curriculum'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
