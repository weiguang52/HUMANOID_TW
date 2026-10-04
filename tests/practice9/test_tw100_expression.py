import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
import torch

ROOT = Path(__file__).parents[2]
PKG = '_tw100_test'
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / 'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp')]
sys.modules[PKG] = pkg
spec = importlib.util.spec_from_file_location(PKG + '.expression', Path(pkg.__path__[0]) / 'expression.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(ROOT / 'scripts/practice9'))
from build_tw100_standing import build


def env_for(c):
    return types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda name: c))


class ExpressionTests(unittest.TestCase):
    def test_upper_targets_ignore_arbitrary_leg_changes(self):
        c = types.SimpleNamespace(cfg=types.SimpleNamespace(joint_names=['leg', 'arm']),
            joint_pos=torch.tensor([[100., 2.]]), joint_vel=torch.tensor([[200., 3.]]),
            robot_joint_vel=torch.tensor([[-200., 4.]]))
        env = env_for(c)
        torch.testing.assert_close(m.upper_command(env, 'motion', ['arm']), torch.tensor([[2.,3.]]))
        self.assertEqual(m.upper_velocity_cost(env, 'motion', ['arm']).item(), 1.)

    def test_contact_observation_preserves_unknown_and_frame_order(self):
        labels = torch.tensor([[1,0],[0,1],[1,1]], dtype=torch.bool)
        known = torch.tensor([[1,1],[0,1],[1,0]], dtype=torch.bool)
        original = labels.clone()
        c = types.SimpleNamespace(_contact_reference=(labels, known), frame_indices=torch.tensor([2,0,1]))
        torch.testing.assert_close(m.contact_command(env_for(c), 'motion'),
            torch.tensor([[1.,1.,1.,0.],[1.,0.,1.,1.],[0.,1.,0.,1.]]))
        torch.testing.assert_close(labels, original)

    def test_treadmill_is_not_standing_drift(self):
        c = types.SimpleNamespace(_expression_standing=torch.tensor([False, True]),
            motion_ids=torch.tensor([0,1]), anchor_lin_vel_w=torch.zeros(2,3),
            anchor_pos_w=torch.ones(2,3), robot_anchor_pos_w=torch.zeros(2,3))
        v = m.standing_drift(env_for(c), 'motion', .08)
        self.assertEqual(v[0].item(),0.)
        self.assertGreater(v[1].item(),.99)

    def test_subset_preserves_contacts_and_equal_clip_mass(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            side = root/'source.npz'
            contact=np.array([[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,0]],dtype=np.uint8)
            np.savez(side, clip_ids=['a','b','c'], lengths=[2,2,3], contact=contact,
                known=1-contact, source_contact4=np.repeat(contact,2,axis=1),
                foot_names=['left_foot','right_foot'],version='test')
            source=root/'source.json'
            source.write_text(json.dumps(dict(motions=[
                dict(id='a',frames=2,category='standing_upper'),
                dict(id='b',frames=2,category='walking'),
                dict(id='c',frames=3,category='standing_upper')],
                contact_reference=dict(file=str(side),sha256=hashlib.sha256(side.read_bytes()).hexdigest()))))
            target=root/'out.json'
            self.assertEqual(build(source,target),2)
            d=json.loads(target.read_text())
            self.assertEqual([r['weight']*(r['frames']-1) for r in d['motions']],[.5,.5])
            with np.load(d['contact_reference']['file']) as a:
                np.testing.assert_array_equal(a['contact'],contact[[0,1,4,5,6]])
                np.testing.assert_array_equal(a['known'],(1-contact)[[0,1,4,5,6]])
            with self.assertRaises(FileExistsError):build(source,target)

if __name__ == '__main__':
    unittest.main()
