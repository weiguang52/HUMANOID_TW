"""TW robot factory. Requires a separately compiled/audited MuJoCo asset."""

import os, math
from pathlib import Path
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from .contract import (
    LEG_MAJOR_JOINTS,
    FOOT_JOINTS,
    WAIST_JOINTS,
    ARM_JOINTS,
    NECK_JOINTS,
)

GROUPS = [
    (LEG_MAJOR_JOINTS, 20.0, 1.0, 40.0, 0.001, 0.5),
    (FOOT_JOINTS, 10.0, 0.5, 15.0, 0.001, 0.375),
    (WAIST_JOINTS, 10.0, 0.5, 20.0, 0.001, 0.5),
    (ARM_JOINTS, 5.0, 0.25, 8.0, 0.0005, 0.4),
    (NECK_JOINTS, 3.0, 0.15, 3.0, 0.0002, 0.25),
]
ACTION_SCALE = {
    name: scale for names, kp, kd, effort, armature, scale in GROUPS for name in names
}


def get_spec():
    path = Path(os.environ["TW_SUPERMIMIC_MJCF"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return mujoco.MjSpec.from_file(str(path))


def get_tw_robot_cfg():
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=tuple(names),
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            armature=armature,
        )
        for names, kp, kd, effort, armature, scale in GROUPS
    )
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.264),
            joint_pos={
                ".*": 0.0,
                "left_shoulder_roll_joint": -math.pi / 2,
                "right_shoulder_roll_joint": math.pi / 2,
            },
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators, soft_joint_pos_limit_factor=1.0
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision.*",),
                contype=0,
                conaffinity=1,
                condim=3,
                friction=(1.0,),
            ),
        ),
    )


# Velocity limits and contact/PD equivalence still require runtime validation.
# This factory is not an accepted training configuration by itself.
