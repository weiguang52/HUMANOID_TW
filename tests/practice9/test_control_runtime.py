import sys,json
from types import SimpleNamespace as NS
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/rsl_rl'))
from control_runtime import save,restore

def cfg():
    return NS(actions=NS(JointPositionAction=NS(smoothing_tau=.04,limit_target_velocity=True,target_velocity_scale=.5)),scene=NS(robot=NS(actuators={'leg_major':NS(stiffness=20.,damping=1.)})))

def test_checkpoint_control_contract_and_explicit_override(tmp_path,monkeypatch):
    original=cfg();save(original,tmp_path)
    replay=cfg();replay.actions.JointPositionAction.smoothing_tau=0
    replay.scene.robot.actuators['leg_major'].damping=99
    restore(replay,tmp_path/'model.pt')
    assert replay.actions.JointPositionAction.smoothing_tau==.04
    assert replay.scene.robot.actuators['leg_major'].damping==1
    monkeypatch.setenv('P9_PD_DAMPING_MULT','2')
    replay.scene.robot.actuators['leg_major'].damping=2
    restore(replay,tmp_path/'model.pt')
    assert replay.scene.robot.actuators['leg_major'].damping==2
