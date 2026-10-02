#!/usr/bin/env python3
"""Run native retarget -> validated Isaac FK -> custom humanoid PPO stages."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from native_retarget import DEFAULT_ROOT
from retarget_humanml3d import CUSTOM_JOINT_NAMES

REPO = Path(__file__).parents[2]
DATA = Path(os.environ.get('GPUFREE_DATA_ROOT', '/root/gpufree-data'))
DEFAULT_OUTPUT = DATA / 'datasets/practice9/tw56_native_v1'


def require_manifest(path, *, training=False):
    payload = json.loads(path.read_text())
    backend = payload.get('backend') or {}
    if backend.get('backend') != 'tw_retargeting_cpp':
        raise ValueError('This pipeline requires a native tw_retargeting manifest')
    if payload.get('joint_names') != CUSTOM_JOINT_NAMES:
        raise ValueError('Manifest uses a different joint coordinate contract')
    motions = payload.get('motions', [])
    if not motions or any(m.get('quality_pass') is not True for m in motions):
        raise ValueError('Manifest must contain quality-passed motions')
    if float(payload['target_fps']) != 50:
        raise ValueError('Training pipeline requires 50 Hz motions')
    for record in motions:
        file = Path(record['file'])
        if not file.is_absolute():
            file = path.parent / file
        if not file.is_file():
            raise FileNotFoundError(file)
        if training and record.get('fk_quality_pass') is not True:
            raise ValueError('Training requires validated FK records')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['build', 'retarget', 'fk', 'train', 'all'], default='all')
    parser.add_argument('--input', type=Path, help='20 Hz NPY, directory, or text list of source paths')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--retarget-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--max-iterations', type=int, default=1)
    parser.add_argument('--run-name', default='tw56_native_smoke')
    args = parser.parse_args()
    if args.stage in ('all', 'retarget') and args.input is None:
        parser.error('--input is required for retarget/all; no implicit full-dataset run')
    if args.num_envs < 1 or args.max_iterations < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Environment, iteration and clip counts must be positive')
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    retarget_manifest = args.output_root / 'retarget_manifest.json'
    training_manifest = args.output_root / 'manifest.json'
    environment = os.environ.copy()
    environment.update(GPUFREE_DATA_ROOT=str(DATA), PRACTICE9_RETARGET_ROOT=str(args.retarget_root.resolve()),
                       PRACTICE9_CUSTOM_MOTION_MANIFEST=str(training_manifest), P9_CUSTOM_GPU=args.gpu,
                       P9_CUSTOM_NUM_ENVS=str(args.num_envs), P9_CUSTOM_MAX_ITERATIONS=str(args.max_iterations),
                       P9_CUSTOM_RUN_NAME=args.run_name, CUDA_VISIBLE_DEVICES=args.gpu,
                       ISAACLAB_PATH=str(DATA / 'projects/IsaacLab'))
    environment['PYTHONPATH'] = str(REPO / 'source/unitree_rl_lab') + ':' + environment.get('PYTHONPATH', '')
    environment['LD_LIBRARY_PATH'] = str(DATA / 'isaacsim') + ':' + environment.get('LD_LIBRARY_PATH', '')

    def run(command):
        print('Running:', ' '.join(str(x) for x in command), flush=True)
        subprocess.run([str(x) for x in command], cwd=REPO, env=environment, check=True)

    if args.stage in ('build', 'all'):
        run(['bash', REPO / 'scripts/practice9/build_native_retarget.sh'])
    if args.stage in ('retarget', 'all'):
        run([sys.executable, REPO / 'scripts/practice9/prepare_custom_robot_urdf.py'])
        command = [sys.executable, REPO / 'scripts/practice9/retarget_humanml3d.py', '--input', args.input,
                   '--output-dir', args.output_root / 'retargeted', '--retarget-root', args.retarget_root,
                   '--continue-on-error']
        if args.limit is not None:
            command += ['--limit', str(args.limit)]
        run(command)
        require_manifest(retarget_manifest)
    if args.stage in ('fk', 'all'):
        require_manifest(retarget_manifest)
        run([sys.executable, REPO / 'scripts/practice9/custom_motion_to_npz.py', '--headless',
             '--input-manifest', retarget_manifest, '--output-dir', args.output_root / 'npz',
             '--output-manifest', training_manifest, '--continue-on-error'])
        require_manifest(training_manifest, training=True)
    if args.stage in ('train', 'all'):
        require_manifest(training_manifest, training=True)
        run(['bash', REPO / 'scripts/practice9/train_humanml3d_custom.sh', '--logger', 'tensorboard'])


if __name__ == '__main__':
    main()
