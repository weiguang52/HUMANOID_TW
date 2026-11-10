"""Meaningful timer/config gates; run remotely with the installed Isaac Lab."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "post_push_unit", ROOT / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/post_push_recovery.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def environment(n=3):
    data = SimpleNamespace(root_quat_w=torch.tensor([quaternion(0,.2,0)]*n),
                           root_ang_vel_b=torch.tensor([[0.,.7,0.]]*n))
    return SimpleNamespace(num_envs=n, device="cpu", step_dt=.02, common_step_counter=10,
                           scene={"robot": SimpleNamespace(data=data)})


def differences(a, b, p=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:
                yield p+"/"+k
            else:
                yield from differences(a[k], b[k], p+"/"+k)
    elif a != b:
        yield p


class TimerTests(unittest.TestCase):
    def test_no_push_is_always_zero(self):
        e=environment()
        for t in [0,10,500000]:
            e.common_step_counter=t
            self.assertTrue(torch.equal(m.pitch_cost(e), torch.zeros(3)))

    def test_window_selected_ids_and_exact_endpoint(self):
        e=environment();m.mark_push(e,torch.tensor([0,2]))
        for t in [10,11,59,60]:
            e.common_step_counter=t
            self.assertGreater(m.pitch_cost(e)[0].item(),0)
            self.assertEqual(m.pitch_cost(e)[1].item(),0)
        e.common_step_counter=61
        self.assertTrue(torch.equal(m.pitch_cost(e),torch.zeros(3)))

    def test_reset_has_no_previous_episode_window(self):
        e=environment();m.mark_push(e,None);m.reset_push_clock(e,torch.tensor([0,2]))
        self.assertEqual(m.pitch_cost(e)[0].item(),0)
        self.assertGreater(m.pitch_cost(e)[1].item(),0)
        m.reset_push_clock(e)
        self.assertTrue(torch.equal(m.pitch_cost(e),torch.zeros(3)))

    def test_repeated_push_only_restarts_selected_windows(self):
        e=environment();m.mark_push(e,None)
        e.common_step_counter=40;m.mark_push(e,torch.tensor([0]))
        e.common_step_counter=70
        self.assertGreater(m.pitch_cost(e)[0].item(),0)
        self.assertEqual(m.pitch_cost(e)[1].item(),0)

    def test_exact_pitch_and_euler_pitch_rate(self):
        e=environment();e.scene["robot"].data.root_quat_w[:]=torch.tensor(quaternion(.4,.2,.7))
        e.scene["robot"].data.root_ang_vel_b[:]=torch.tensor([.3,.7,-.2])
        m.mark_push(e,None)
        expected=.2**2+.1*(math.cos(.4)*.7-math.sin(.4)*(-.2))**2
        torch.testing.assert_close(m.pitch_cost(e),torch.full((3,),expected),rtol=1e-5,atol=1e-7)

    def test_default_off_does_not_mutate_any_config(self):
        e=SimpleNamespace(events="sentinel",observations={"policy_dim":100},rewards=None)
        before=copy.deepcopy(vars(e));m.configure_post_push_pitch_reward(e,None)
        self.assertEqual(vars(e),before)

    def test_invalid_weight_or_nonintegral_window_rejected(self):
        for w in [-.1,float("nan"),float("inf")]:
            with self.assertRaises(ValueError):m.configure_post_push_pitch_reward(None,w)
        for window in [0,.001,1.005]:
            with self.assertRaises(ValueError):m.push_window(environment(),window)


class MonitorTests(unittest.TestCase):
    def test_monitor_flush_outside_inference_mode_and_accumulator_reset(self):
        import tempfile
        sys.path.insert(0,str(ROOT / "scripts/locomotion"))
        from issue12_monitor import RewardMonitor
        e=environment();m.mark_push(e,None)
        manager=SimpleNamespace(active_terms=[m.TERM,"track_lin_vel_xy","track_ang_vel_z"],
                                _step_reward=torch.zeros((3,3)))
        def compute(dt):
            manager._step_reward[:,0]=-.7*m.pitch_cost(e)
            manager._step_reward[:,1]=3.
            manager._step_reward[:,2]=1.
            return manager._step_reward.sum(-1)*dt
        manager.compute=compute;e.reward_manager=manager
        with tempfile.TemporaryDirectory() as path:
            monitor=RewardMonitor(e,.7,path)
            for i in range(2):
                with torch.inference_mode():manager.compute(.02)
                row=monitor.flush(i)
                self.assertEqual(row["environment_steps"],1)
                self.assertEqual(row["transitions"],3)
                self.assertAlmostEqual(row["new_reward_per_step"],-.7*(.2**2+.1*.7**2)*.02,places=7)
            monitor.close()


class ConfigTests(unittest.TestCase):
    def test_policy_critic_and_action_configs_unchanged(self):
        from unitree_rl_lab.tasks.locomotion.robots.tw.velocity_env_cfg import RobotEnvCfg
        cfg=RobotEnvCfg()
        obs=copy.deepcopy(cfg.observations.to_dict());actions=copy.deepcopy(cfg.actions.to_dict())
        m.configure_post_push_pitch_reward(cfg,.7)
        self.assertEqual(cfg.observations.to_dict(),obs)
        self.assertEqual(cfg.actions.to_dict(),actions)

    def test_matched_arms_differ_only_in_one_reward_weight(self):
        from unitree_rl_lab.tasks.locomotion.robots.tw.velocity_env_cfg import RobotEnvCfg
        a=RobotEnvCfg();b=copy.deepcopy(a)
        m.configure_post_push_pitch_reward(a,0.)
        m.configure_post_push_pitch_reward(b,.7)
        self.assertEqual(list(differences(a.to_dict(),b.to_dict())),
                         ["/rewards/post_push_pitch_recovery/weight"])


if __name__=="__main__":
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True).app
    result=unittest.main(verbosity=2,exit=False)
    app.close()
    sys.exit(0 if result.result.wasSuccessful() else 1)
