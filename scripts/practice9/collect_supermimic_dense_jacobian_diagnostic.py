"""Explicit dense-Jacobian collection diagnostic; never changes training defaults."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import runpy
import sys
import mujoco
from supermimic.config.tw import env_cfgs, velocity_limits

output=Path(sys.argv[sys.argv.index('--output')+1]);assert not output.exists()
original_cfg=env_cfgs.tw_supermimic_env_cfg
original_enable=velocity_limits.enable_tw_velocity_limits
record=dict(diagnostic_only=True,training_approved=False,promotion_approved=False,numerical_variant='dense_jacobian_only',wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def dense_cfg(*args,**kwargs):
    cfg=original_cfg(*args,**kwargs)
    before=asdict(cfg.sim.mujoco)
    cfg.sim.mujoco.jacobian='dense'
    after=asdict(cfg.sim.mujoco)
    assert [k for k in before if before[k]!=after[k]]==['jacobian']
    record.update(mujoco_before=before,mujoco_after=after)
    return cfg


def checked_enable(env):
    assert env.sim.mj_model.opt.jacobian==mujoco.mjtJacobian.mjJAC_DENSE
    assert not env.sim.wp_model.is_sparse
    record.update(runtime_dense_verified=True,nv=int(env.sim.mj_model.nv))
    return original_enable(env)


env_cfgs.tw_supermimic_env_cfg=dense_cfg
velocity_limits.enable_tw_velocity_limits=checked_enable
try:
    runpy.run_path(str(Path(__file__).with_name('collect_supermimic_motion_pd.py')),run_name='__main__')
finally:
    env_cfgs.tw_supermimic_env_cfg=original_cfg
    velocity_limits.enable_tw_velocity_limits=original_enable
    if output.exists():
        (output/'dense_jacobian_diagnostic.json').write_text(json.dumps(record,indent=2))
assert record.get('runtime_dense_verified'), 'Dense runtime was not verified'
