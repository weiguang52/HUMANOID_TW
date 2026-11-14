"""Sequential issue19 preflight; no formal training unless every gate passes."""
import json, os, subprocess, time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def main():
 os.chdir(REPO)
 while not (ROOT/'actor_pretraining_validation.json').exists():
  raise RuntimeError('Fixed-state/ONNX pretraining validation must pass first')
 assert json.loads((ROOT/'actor_pretraining_validation.json').read_text())['passed']
 record=next(r for r in json.loads((ROOT.with_name('issue17_20261113')/'training_records.json').read_text()) if r['group']=='sym_int' and r['training_seed']==1)
 checkpoint=Path(record['checkpoint'])
 for mode in ['baseline','sym_cont','sym_wrap']:
  out=ROOT/'preflight'/mode;out.mkdir(parents=True,exist_ok=True)
  assert not (out/'final.pt').exists()
  args=['python','-u','scripts/locomotion/preflight_train_issue19.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','4096','--max_iterations','2','--run_name','issue19_preflight_'+mode,'--seed','1','--resume','--load_run',checkpoint.parent.name,'--checkpoint',checkpoint.name,'--experiment_profile','robust08','--issue5_training_mode','strengthen','--issue5_factor','push','--mass_randomization','bounded','--last_action_observation','requested','--post_push_pitch_weight','0.0','--issue12_audit_dir',str(out),'--issue17_mode','sym_int']
  if mode!='baseline':args+=['--issue19_mode',mode]
  env=os.environ.copy();env['TW_ISSUE19_PREFLIGHT_MODE']=mode
  with (out/'console.log').open('w') as log:
   p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue19',*args],stdout=log,stderr=subprocess.STDOUT,env=env)
   (ROOT/'preflight_progress.json').write_text(json.dumps({'mode':mode,'child_pid':p.pid,'started':time.time()}))
   assert p.wait()==0,mode
  assert (out/'final.pt').exists()
  print('SMOKE_TWO_UPDATES_DONE',mode,flush=True)
 import torch,numpy as np
 def equal(a,b,path=''):
  assert type(a)==type(b),(path,type(a),type(b))
  if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
  elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
  elif isinstance(a,dict):
   assert a.keys()==b.keys(),path
   for k in a:equal(a[k],b[k],path+'/'+str(k))
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b),path
   for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
  else:assert a==b,path
 for name in ['final.pt','runtime.pt']:
  equal(torch.load(ROOT/'preflight/baseline'/name,weights_only=False,map_location='cpu'),torch.load(ROOT/'preflight/sym_cont'/name,weights_only=False,map_location='cpu'),name)
 wrapped=torch.load(ROOT/'preflight/sym_wrap/final.pt',weights_only=False,map_location='cpu')
 assert all(torch.isfinite(t).all() for t in wrapped['actor_state_dict'].values())
 summary=json.loads((ROOT/'preflight/sym_wrap/run_summary.json').read_text());assert summary['additional_updates']==2 and summary['environment_steps']==48
 result={'passed':True,'disabled_model_optimizer_and_runtime_bitwise_equal':True,'sym_wrap_two_updates':True,'all_three_runs_4096_envs':True,'summary':summary}
 (ROOT/'preflight_validation.json').write_text(json.dumps(result,indent=2))
 print('ALL_ISSUE19_PREFLIGHT_GATES_PASS',flush=True)
if __name__=='__main__':main()
