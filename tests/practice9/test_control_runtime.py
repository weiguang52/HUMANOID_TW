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


def test_position_limit_roundtrip_and_legacy(tmp_path):
    original=cfg();original.actions.JointPositionAction.limit_target_position=True
    save(original,tmp_path)
    replay=cfg();restore(replay,tmp_path/'model.pt')
    assert replay.actions.JointPositionAction.limit_target_position is True
    path=tmp_path/'params/control_runtime.json'
    payload=json.loads(path.read_text());payload['version']=1
    del payload['limit_target_position'];path.write_text(json.dumps(payload))
    restore(replay,tmp_path/'model.pt')
    assert replay.actions.JointPositionAction.limit_target_position is False


def test_old_waist_checkpoint_cannot_load_in_new_robot(tmp_path):
    import pytest
    original=cfg();save(original,tmp_path)
    replay=cfg();replay.actions.JointPositionAction.joint_names=['waist_pitch_joint']
    with pytest.raises(ValueError,match='Legacy checkpoint'):
        restore(replay,tmp_path/'model.pt')
    original.actions.JointPositionAction.joint_names=['waist_pitch_joint']
    save(original,tmp_path);restore(replay,tmp_path/'model.pt')
    replay.actions.JointPositionAction.joint_names=['chest_pitch']
    with pytest.raises(ValueError,match='coordinate names'):
        restore(replay,tmp_path/'model.pt')
