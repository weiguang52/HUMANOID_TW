"""Real ActionManager tests for requested/applied history under delay/reset."""
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import unittest,json,copy
from pathlib import Path
from types import SimpleNamespace as NS
import torch,gymnasium as gym
import isaaclab_tasks,unitree_rl_lab.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import mdp
from unitree_rl_lab.tasks.locomotion.robots.tw.experiments import configure_experiment,configure_issue5_training
from unitree_rl_lab.tasks.locomotion.robots.tw.mass_randomization import configure_mass_randomization
from unitree_rl_lab.tasks.locomotion.robots.tw.action_observation import configure_action_observation
ROOT=Path('/root/gpufree-data/tw_stage1/issue8_20261109')
cfg=parse_env_cfg('Unitree-TW-30dof-Velocity',device='cuda:0',num_envs=64)
configure_experiment(cfg,NS(algorithm=NS()),'robust08')
configure_issue5_training(cfg,'strengthen','push')
configure_mass_randomization(cfg,'bounded')
critic_before=copy.deepcopy(cfg.observations.critic.last_action.params)
requested_before=copy.deepcopy(cfg.observations.policy.last_action.params)
configure_action_observation(cfg,'requested')
assert cfg.observations.policy.last_action.params==requested_before
configure_action_observation(cfg,'applied')
assert cfg.observations.critic.last_action.params==critic_before
env=gym.make('Unitree-TW-30dof-Velocity',cfg=cfg).unwrapped
term=env.action_manager.get_term('JointPositionAction')
def observed():return mdp.last_action(env,**cfg.observations.policy.last_action.params)
def action(value):return torch.full((64,30),value,device=env.device)

class ActionObservationTests(unittest.TestCase):
    def setUp(self):env.reset()
    def test_zero_delay_observations_equal(self):
        term.delay.fill_(False)
        for value in [.1,.3,-.2]:
            env.action_manager.process_action(action(value))
            self.assertTrue(torch.equal(observed(),mdp.last_action(env)))
            self.assertEqual(tuple(observed().shape),(64,30))
    def test_twenty_ms_observation_matches_applied_target(self):
        term.delay.fill_(True);previous=action(0.)
        for value in [.1,.3,-.2]:
            current=action(value);env.action_manager.process_action(current)
            self.assertTrue(torch.equal(observed(),previous))
            self.assertTrue(torch.equal(mdp.last_action(env),current))
            self.assertTrue(torch.equal(observed(),term.raw_actions))
            term.apply_actions()
            robot=env.scene['robot']
            self.assertTrue(torch.equal(robot.data.joint_pos_target[:,term._joint_ids],term.processed_actions))
            previous=current.clone()
    def test_full_and_partial_reset_clear_history(self):
        term.delay.fill_(True)
        env.action_manager.process_action(action(.2));env.action_manager.process_action(action(.4))
        ids=torch.tensor([0,2],device=env.device);env._reset_idx(ids)
        self.assertFalse(term.previous[ids].any());self.assertFalse(observed()[ids].any())
        self.assertTrue(torch.equal(term.previous[1],action(.4)[1]))
        term.delay.fill_(True);env.action_manager.process_action(action(.6))
        self.assertFalse(observed()[ids].any())
        self.assertTrue(torch.equal(observed()[1],action(.4)[1]))
        env.reset();self.assertFalse(term.previous.any());self.assertFalse(observed().any())
    def test_policy_parameter_only_and_dimension_unchanged(self):
        self.assertEqual(cfg.observations.policy.last_action.params,{'action_name':'JointPositionAction'})
        self.assertEqual(cfg.observations.critic.last_action.params,critic_before)
        self.assertEqual(tuple(mdp.last_action(env).shape),tuple(observed().shape))
suite=unittest.defaultTestLoader.loadTestsFromTestCase(ActionObservationTests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
ROOT.mkdir(exist_ok=True)
(ROOT/'action_observation_tests.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'policy_params':cfg.observations.policy.last_action.params,'critic_params':critic_before,'observation_dims':env.observation_manager.group_obs_dim,'action_shape':list(observed().shape)},indent=2))
env.close();app.close()
raise SystemExit(0 if result.wasSuccessful() else 1)
