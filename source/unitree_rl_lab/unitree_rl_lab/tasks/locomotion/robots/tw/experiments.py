"""Opt-in issue #2/#3 ablations; the production task defaults are unchanged."""
import torch
from isaaclab.envs.mdp.actions import JointPositionAction


class RandomOneStepDelayAction(JointPositionAction):
    """Uniform per-episode 0/1 control-step target delay (0/20 ms)."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros_like(self._raw_actions)
        self.delay = torch.randint(0, 2, (env.num_envs, 1), device=env.device).bool()

    def process_actions(self, actions):
        delayed = torch.where(self.delay, self.previous, actions)
        self.previous.copy_(actions)
        super().process_actions(delayed)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, "previous"):
            ids = slice(None) if env_ids is None else env_ids
            self.previous[ids] = 0.
            self.delay[ids] = torch.randint(0, 2, self.delay[ids].shape, device=self.delay.device).bool()


def configure_experiment(env_cfg, agent_cfg, profile):
    if profile == "default":
        return
    valid = {"control", "entropy", "yaw", "range05", "range08", "robust05", "robust08"}
    if profile not in valid:
        raise ValueError(profile)
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (-.1, .3)
    cmd.ranges.lin_vel_y = (-.1, .1)
    cmd.ranges.ang_vel_z = (-.3, .3)
    if profile != "control":
        agent_cfg.algorithm.entropy_coef = .001
    if profile in {"yaw", "range05", "range08", "robust05", "robust08"}:
        env_cfg.rewards.track_ang_vel_z.weight = 1.0
    if profile in {"range05", "range08", "robust05", "robust08"}:
        vmax = .8 if profile.endswith("08") else .5
        yawmax = .8 if profile.endswith("08") else .5
        cmd.ranges.lin_vel_x = (-.1, vmax)
        cmd.limit_ranges.lin_vel_x = (-.1, vmax)
        cmd.ranges.ang_vel_z = (-yawmax, yawmax)
        cmd.limit_ranges.ang_vel_z = (-yawmax, yawmax)
    if profile.startswith("robust"):
        env_cfg.events.push_robot.params["velocity_range"] = {"x":(-.3,.3),"y":(-.3,.3)}
        env_cfg.events.physics_material.params["static_friction_range"] = (.4,1.4)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (.3,1.2)
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (-.08,.15)
        env_cfg.actions.JointPositionAction.class_type = RandomOneStepDelayAction
        # Observation corruption already exists; retain its archived noise ranges.
