"""TW-specific configuration; cross-simulator equivalence is not assumed."""

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from supermimic.config.g1.env_cfgs import unitree_g1_supermimic_env_cfg
from supermimic.config.g1.rl_cfg import unitree_g1_supermimic_cfg
from .robot import get_tw_robot_cfg, ACTION_SCALE
from .contract import CUSTOM_HUMANOID_TRACKING_BODY_NAMES


def tw_supermimic_env_cfg(motion_files, num_envs=1024, play=False):
    cfg = unitree_g1_supermimic_env_cfg(motion_files, num_envs=num_envs, play=play)
    cfg.scene.entities["robot"] = get_tw_robot_cfg()
    # Actual TW mesh contacts exceed the G1 buffer (59 at initial assembly).
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
    cfg.viewer.body_name = "chest"
    cfg.viewer.distance = 1.0
    cfg.actions["joint_pos"].scale = ACTION_SCALE
    cfg.commands["motion"].body_names = tuple(CUSTOM_HUMANOID_TRACKING_BODY_NAMES)
    cfg.commands["motion"].anchor_body_name = "chest"
    cfg.commands["motion"].require_joint_name_mapping = True
    cfg.commands["motion"].required_coordinate_contract = "tw123_yaw_waist_v3"
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("chest",)
    )
    cfg.events["base_com"].params["ranges"] = {i: (-0.002, 0.002) for i in range(3)}
    cfg.events["encoder_bias"].params["bias_range"] = (-0.005, 0.005)
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=r"^(left|right)_foot_collision(_\d+)?$"
    )
    cfg.events["foot_friction"].params["ranges"] = (0.6, 1.1)
    cfg.terminations["anchor_pos"].params["threshold"] = 0.12
    cfg.terminations["ee_body_pos"] = TerminationTermCfg(
        func=tracking_mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.12,
            "body_names": ["left_foot", "right_foot", "left_wrist", "right_wrist"],
        },
    )
    cfg.episode_length_s = 30.0
    return cfg


def tw_supermimic_cfg():
    cfg = unitree_g1_supermimic_cfg()
    cfg.context_torso_body_name = "chest"
    cfg.context_left_foot_geom_pattern = r"^left_foot_collision(_\d+)?$"
    cfg.context_right_foot_geom_pattern = r"^right_foot_collision(_\d+)?$"
    # Carry existing TW core metric scales into the SuperMimic reward form.
    cfg.reward_body_pos_std = 0.15
    cfg.reward_root_pos_std = 0.15
    cfg.reward_body_ori_std = 0.5
    cfg.reward_root_ori_std = 0.4
    cfg.reward_body_lin_vel_std = 0.75
    cfg.reward_body_ang_vel_std = 2.0
    cfg.experiment_name = "tw_supermimic"
    return cfg
