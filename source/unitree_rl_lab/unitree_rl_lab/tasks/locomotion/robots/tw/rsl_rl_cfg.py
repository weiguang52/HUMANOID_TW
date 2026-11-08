from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class TWPPORunnerCfg(BasePPORunnerCfg):
    """TW is a ~0.84 kg robot with weak PD gains, so exploration noise must start small.

    With init_noise_std=1.0 and action scales of 0.25-0.5 rad, random targets collapse the robot
    within ~13 control steps before PPO can learn anything.
    """

    max_iterations = 3000
    experiment_name = "tw_stage1"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # Iteration 2: with entropy_coef=0.01 the action std grew back after iter ~600 and the policy degraded
    # (time_out 95% -> 58%). Remove the entropy bonus; everything else identical to BasePPORunnerCfg.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
