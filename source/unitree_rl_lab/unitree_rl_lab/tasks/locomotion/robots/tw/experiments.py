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


class FixedOneStepDelayAction(RandomOneStepDelayAction):
    """Exactly one 20 ms control-step delay, for isolated evaluation."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.delay.fill_(True)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, "delay"):
            self.delay[slice(None) if env_ids is None else env_ids] = True


def configure_issue5_condition(env_cfg, condition):
    """Independent factors; called after the archived robust08 profile."""
    if condition == "legacy":
        return
    valid = {"clean", "push", "material", "delay", "noise", "combined"}
    if condition not in valid:
        raise ValueError(condition)
    if condition not in {"push", "combined"}:
        env_cfg.events.push_robot = None
    if condition not in {"material", "combined"}:
        env_cfg.events.physics_material = None
        env_cfg.events.add_base_mass = None
    env_cfg.observations.policy.enable_corruption = condition in {"noise", "combined"}
    if condition == "delay":
        env_cfg.actions.JointPositionAction.class_type = FixedOneStepDelayAction
    elif condition != "combined":
        env_cfg.actions.JointPositionAction.class_type = JointPositionAction


class RandomTwoStepDelayAction(JointPositionAction):
    """Uniform 0/20/40 ms per-episode target latency for the strengthened arm."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros_like(self._raw_actions)
        self.older = torch.zeros_like(self._raw_actions)
        self.lag = torch.randint(0, 3, (env.num_envs, 1), device=env.device)

    def process_actions(self, actions):
        delayed = torch.where(self.lag == 0, actions,
                              torch.where(self.lag == 1, self.previous, self.older))
        self.older.copy_(self.previous)
        self.previous.copy_(actions)
        super().process_actions(delayed)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, "previous"):
            ids = slice(None) if env_ids is None else env_ids
            self.previous[ids] = 0.
            self.older[ids] = 0.
            self.lag[ids] = torch.randint(0, 3, self.lag[ids].shape, device=self.lag.device)


def configure_issue5_training(env_cfg, mode, factor):
    """Change only the selected component of the original robust08 package."""
    if mode == "legacy":
        return
    if mode not in {"remove", "strengthen"} or factor not in {"push", "material", "delay", "noise"}:
        raise ValueError((mode, factor))
    if factor == "push":
        if mode == "remove":
            env_cfg.events.push_robot = None
        else:
            env_cfg.events.push_robot.params["velocity_range"] = {"x":(-.45,.45), "y":(-.45,.45)}
    elif factor == "material":
        if mode == "remove":
            env_cfg.events.physics_material = None
            env_cfg.events.add_base_mass = None
        else:
            env_cfg.events.physics_material.params["static_friction_range"] = (.3,1.6)
            env_cfg.events.physics_material.params["dynamic_friction_range"] = (.2,1.3)
            env_cfg.events.add_base_mass.params["mass_distribution_params"] = (-.10,.20)
    elif factor == "delay":
        env_cfg.actions.JointPositionAction.class_type = (JointPositionAction if mode == "remove" else RandomTwoStepDelayAction)
    elif mode == "remove":
        env_cfg.observations.policy.enable_corruption = False
    else:
        for term in vars(env_cfg.observations.policy).values():
            noise = getattr(term, "noise", None)
            if noise is not None and hasattr(noise, "n_min") and hasattr(noise, "n_max"):
                noise.n_min *= 2.
                noise.n_max *= 2.
