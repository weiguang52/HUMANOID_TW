"""Expand replay-verified root velocity levels; diagnostic physics only."""
import ast,hashlib,json,sys
from pathlib import Path
from dataclasses import asdict
import mujoco
from supermimic.config.tw import env_cfgs,velocity_limits
source=Path(__file__).with_name('collect_supermimic_motion_root_velocity.py');levels=[1.,2.,8.,32.,64.]
output=Path(sys.argv[sys.argv.index('--output')+1]);assert not output.exists()
tree=ast.parse(source.read_text());plans=[];loops=[]
for n in ast.walk(tree):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='contract' and isinstance(k.value,ast.Constant) and k.value.value=='motion_root_velocity_branches_v1' for k in n.keywords):
  plans.append(n)
  for k in n.keywords:
   if k.arg=='levels':k.value=ast.parse(repr(levels),mode='eval').body
   if k.arg=='contract':k.value=ast.Constant('wide_motion_root_velocity_branches_v1')
 if isinstance(n,ast.For) and ast.unparse(n.target)=='(level, assignment)':
  loops.append(n);n.iter=ast.parse(repr([(0.,0)]+[(level,i) for level in levels for i in [0,1]]),mode='eval').body
assert len(plans)==len(loops)==1
original_cfg=env_cfgs.tw_supermimic_env_cfg;original_enable=velocity_limits.enable_tw_velocity_limits
record={'diagnostic_only':True,'training_approved':False,'promotion_approved':False,'levels':levels,'base_collector_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selection':'Levels1,2 retained; add8,32,64 to cover fit-only maximum trajectory-median p95 from hidden_moment_root_velocity_exposure_v1; no validation fitting'}
def dense_cfg(*args,**kwargs):
 cfg=original_cfg(*args,**kwargs);before=asdict(cfg.sim.mujoco);cfg.sim.mujoco.jacobian='dense';after=asdict(cfg.sim.mujoco);assert [k for k in before if before[k]!=after[k]]==['jacobian'];record.update(mujoco_before=before,mujoco_after=after);return cfg
def checked_enable(env):
 assert env.sim.mj_model.opt.jacobian==mujoco.mjtJacobian.mjJAC_DENSE and not env.sim.wp_model.is_sparse;record['runtime_dense_verified']=True;return original_enable(env)
env_cfgs.tw_supermimic_env_cfg=dense_cfg;velocity_limits.enable_tw_velocity_limits=checked_enable
try:exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
finally:
 env_cfgs.tw_supermimic_env_cfg=original_cfg;velocity_limits.enable_tw_velocity_limits=original_enable
 if output.exists():(output/'wide_velocity_diagnostic.json').write_text(json.dumps(record,indent=2))
assert record.get('runtime_dense_verified')
