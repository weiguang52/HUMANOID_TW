"""Adapter for tw_retargeting's selected native C++ backend; no legacy solver."""
from __future__ import annotations
import hashlib
import json
import os
import re
import subprocess
import tempfile
import sysconfig
from pathlib import Path
import numpy as np

NATIVE_FPS = 90.0
NATIVE_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_pitch_joint', 'left_ankle_yaw_joint', 'left_ankle_pitch_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_pitch_joint', 'right_ankle_yaw_joint', 'right_ankle_pitch_joint',
    'waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint', 'right_wrist_yaw_joint',
    'neck_yaw_joint', 'neck_roll_joint', 'neck_pitch_joint',
]
DEFAULT_ROOT = Path(os.environ.get('GPUFREE_DATA_ROOT', '/root/gpufree-data')) / 'projects/tw_retargeting/IKRetargeting_V3'


def resample_native(values, output_fps):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 28 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f'Native output must be finite [T>=2,28], got {values.shape}')
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError('Output FPS must be finite and positive')
    duration = (len(values) - 1) / NATIVE_FPS
    count = int(np.floor(duration * output_fps + 1e-8)) + 1
    if count < 3:
        raise ValueError('At least three output frames required for derivatives')
    source_t = np.arange(len(values)) / NATIVE_FPS
    target_t = np.arange(count) / output_fps
    return np.column_stack([np.interp(target_t, source_t, values[:, joint]) for joint in range(28)])


class NativeRetargeter:
    def __init__(self, root=DEFAULT_ROOT, binary=None, timeout=120, profile=None):
        self.root = Path(root).resolve()
        self.binary = Path(binary).resolve() if binary else self.root / 'build/native/tw_retarget'
        self.urdf = self.root / 'assets/urdf/Assembly.urdf'
        config = Path(profile).resolve() if profile is not None else self.root / 'configs/selected_head_axis.json'
        if not self.binary.is_file():
            raise FileNotFoundError(f'Build the native backend first: {self.binary}')
        self.timeout = float(timeout)
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('Native timeout must be positive')
        self.flags = json.loads(config.read_text())['flags']
        if not isinstance(self.flags, list) or not all(isinstance(x, str) for x in self.flags):
            raise ValueError('Selected flags must be a list of strings')
        source = (self.root / 'src/robot.cpp').read_text()
        match = re.search(r'static const std::vector<std::string> names\s*=\s*\{(.*?)\};', source, re.S)
        if match is None or re.findall(r'"([^"]+)"', match.group(1)) != NATIVE_JOINT_NAMES:
            raise ValueError('Upstream joint order changed; update adapter explicitly')
        commit = subprocess.run(['git', '-C', str(self.root), 'rev-parse', 'HEAD'],
                                check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        self.provenance = {
            'backend': 'tw_retargeting_cpp', 'upstream_commit': commit,
            'native_fps': NATIVE_FPS, 'native_joint_names': NATIVE_JOINT_NAMES,
            'binary_sha256': hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            'selected_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
            'solver_urdf_sha256': hashlib.sha256(self.urdf.read_bytes()).hexdigest(),
            'selected_flags': self.flags,
        }
        self.environment = os.environ.copy()
        libraries = [self.root / 'build/deps/cmeel.prefix/lib',
                     Path(sysconfig.get_paths()['purelib']) / 'cmeel.prefix/lib']
        self.environment['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in libraries) + ':' + self.environment.get('LD_LIBRARY_PATH', '')

    def process(self, joints, output_fps):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 3 or joints.shape[1] < 22 or joints.shape[2] != 3 or len(joints) < 9:
            raise ValueError('Input must be [T>=9,J>=22,3] at 20 Hz')
        if not np.isfinite(joints).all():
            raise ValueError('Input contains NaN or Inf')
        tmp_root = Path(os.environ.get('GPUFREE_DATA_ROOT', '/root/gpufree-data')) / 'tmp/tw56_native'
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            source = Path(tmp) / 'motion.npy'
            output = Path(tmp) / 'out'
            np.save(source, np.ascontiguousarray(joints), allow_pickle=False)
            command = [str(self.binary), '--urdf', str(self.urdf), '--input', str(source),
                       '--out', str(output), '--mode', 'stream', '--iterations', '1'] + self.flags
            result = subprocess.run(command, env=self.environment, capture_output=True,
                                    text=True, timeout=self.timeout)
            if result.returncode:
                raise RuntimeError(f'Native retarget failed ({result.returncode}): {result.stderr[-3000:]}')
            values = np.load(output / 'motion.npy', allow_pickle=False)
            timing = json.loads((output / 'timing.json').read_text())
            residual = float(timing.get('max_kkt', float('inf')))
            if not np.isfinite(residual) or residual > 1e-6:
                raise ValueError(f'Native QP residual exceeds contract: {residual}')
            return resample_native(values, output_fps), timing
