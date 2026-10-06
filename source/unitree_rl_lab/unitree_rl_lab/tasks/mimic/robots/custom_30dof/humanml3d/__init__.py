import gymnasium as gym

gym.register(
    id="Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id='Unitree-Custom-Humanoid-30dof-Expression-HumanML3D',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.expression_env_cfg:ExpressionEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.expression_env_cfg:ExpressionPlayEnvCfg',
        'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg',
    },
)

for name in ('Bounded', 'Balanced'):
    gym.register(
        id=f'Unitree-Custom-Humanoid-30dof-{name}-HumanML3D',
        entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': f'{__name__}.balanced_env_cfg:{name}ExpressionEnvCfg',
            'play_env_cfg_entry_point': f'{__name__}.balanced_env_cfg:{name}ExpressionPlayEnvCfg',
            'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg',
        },
    )

for name in ('Step', 'StepVelocity'):
    gym.register(
        id=f'Unitree-Custom-Humanoid-30dof-{name}-HumanML3D',
        entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': f'{__name__}.step_env_cfg:{name}EnvCfg',
            'play_env_cfg_entry_point': f'{__name__}.step_env_cfg:{name}PlayEnvCfg',
            'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg',
        },
    )

for name in ('Size', 'SizeResidual'):
    gym.register(
        id=f'Unitree-Custom-Humanoid-30dof-{name}-HumanML3D',
        entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': f'{__name__}.size_env_cfg:{name}EnvCfg',
            'play_env_cfg_entry_point': f'{__name__}.size_env_cfg:{name}PlayEnvCfg',
            'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg',
        },
    )

for name in ('Path', 'PathRate'):
    gym.register(
        id=f'Unitree-Custom-Humanoid-30dof-{name}-HumanML3D',
        entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': f'{__name__}.path_env_cfg:{name}EnvCfg',
            'play_env_cfg_entry_point': f'{__name__}.path_env_cfg:{name}PlayEnvCfg',
            'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg',
        },
    )

gym.register(
    id='Unitree-Custom-Humanoid-30dof-Knee-HumanML3D',
    entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.knee_env_cfg:KneeEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.knee_env_cfg:KneePlayEnvCfg',
        'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg',
    },
)

gym.register(
    id='Unitree-Custom-Humanoid-30dof-Sole-HumanML3D',
    entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.sole_env_cfg:SoleEnvCfg',
        'play_env_cfg_entry_point': f'{__name__}.sole_env_cfg:SolePlayEnvCfg',
        'rsl_rl_cfg_entry_point': 'unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg',
    },
)
