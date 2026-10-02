import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / 'scripts/practice9'))
from native_retarget import DEFAULT_ROOT, NATIVE_JOINT_NAMES, NativeRetargeter, resample_native
from retarget_humanml3d import validate_humanml3d, restore_root_trajectory, iter_inputs
from run_native_motion_pipeline import require_manifest, CUSTOM_JOINT_NAMES, CONTRACT_VERSION


class NativeAdapterTests(unittest.TestCase):
    def test_resampling_preserves_physical_timestamps(self):
        time = np.arange(91) / 90
        source = np.repeat(time[:, None], 28, axis=1)
        actual = resample_native(source, 50)
        self.assertEqual(actual.shape, (51, 28))
        np.testing.assert_allclose(actual[:, 0], np.arange(51) / 50)
        shorter = resample_native(source[:-1], 50)
        self.assertEqual(shorter.shape, (50, 28))
        self.assertAlmostEqual(shorter[-1, 0], .98)

    def test_invalid_native_output_is_rejected(self):
        for shape in [(10, 27), (1, 28)]:
            with self.assertRaises(ValueError):
                resample_native(np.zeros(shape), 50)
        with self.assertRaises(ValueError):
            resample_native(np.full((10, 28), np.nan), 50)
        with self.assertRaises(ValueError):
            resample_native(np.zeros((10, 28)), 0)

    def test_extended_human_input_and_root_timestamps(self):
        source = np.zeros((21, 43, 3))
        source[:, 0, 2] = np.arange(21) / 20
        source[:, 2, 0] = -1
        validate_humanml3d(source, Path('extended.npy'))
        root, _ = restore_root_trajectory(source, 50, 1., .28, 50.)
        np.testing.assert_allclose(np.linalg.norm(root[:, :2], axis=1), np.arange(50) / 50)
        with self.assertRaises(ValueError):
            validate_humanml3d(np.zeros((21, 21, 3)), Path('short.npy'))

    def test_input_list_rejects_duplicate_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'clip.npy').touch()
            listing = path / 'inputs.txt'
            listing.write_text('clip.npy\n')
            self.assertEqual(iter_inputs(listing, '*.npy', None), [path / 'clip.npy'])
            listing.write_text('clip.npy\nclip.npy\n')
            with self.assertRaisesRegex(ValueError, 'unique'):
                iter_inputs(listing, '*.npy', None)

    def test_pipeline_rejects_legacy_and_quality_failed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps({'backend': None}))
            with self.assertRaisesRegex(ValueError, 'native'):
                require_manifest(path)
            path.write_text(json.dumps({'backend': {'backend': 'tw_retargeting_cpp'},
                                        'joint_coordinate_contract': CONTRACT_VERSION, 'joint_names': CUSTOM_JOINT_NAMES, 'motions': []}))
            with self.assertRaisesRegex(ValueError, 'quality'):
                require_manifest(path)

    @unittest.skipUnless((DEFAULT_ROOT / 'build/native/tw_retarget').is_file(), 'Native binary required')
    def test_native_real_clip_extended_input_matches_standard(self):
        source = Path('/root/gpufree-data/datasets/practice9/humanml3d_rebuild/staging-v1/HumanML3D/new_joints/001512.npy')
        if not source.is_file():
            self.skipTest('HumanML3D sample required')
        backend = NativeRetargeter()
        joints = np.load(source)
        first, timing = backend.process(joints, 50)
        extended = np.zeros((len(joints), 43, 3))
        extended[:, :22] = joints[:, :22]
        second, _ = backend.process(extended, 50)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape[1], len(NATIVE_JOINT_NAMES))
        self.assertLess(timing['max_kkt'], 1e-6)


if __name__ == '__main__':
    unittest.main()
