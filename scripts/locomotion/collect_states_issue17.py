"""Prespecified102400 raw post-step states, verified against committed control rollouts."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113');REPO=Path('/root/gpufree-data/projects/HUMANOID_TW');BASELINE=REPO/'validation_artifacts/stage1_issue15_20261111'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.chdir(REPO);assert not (ROOT/'fixed_state_set.json').exists();cohort=json.loads((ROOT/'cohort.json').read_text());manifest=json.loads((BASELINE/'sha256.json').read_text());files=[]
 for seed in range(1,6):
  assert not (ROOT/'stop').exists()
  # Avoid interfering with any formal training already running.
  while (ROOT/'training_progress.json').exists() and json.loads((ROOT/'training_progress.json').read_text()).get('state')=='training' and not (ROOT/'formal_training_complete').exists():time.sleep(20)
  model=cohort['models'][f'control6000_s{seed}'];assert sha(model['checkpoint'])==model['sha256'];name=f'control6000_s{seed}_eval1';folder=ROOT/'state_collection';folder.mkdir(exist_ok=True);output=folder/(name+'.json');states=folder/(name+'_states.npz');moments=folder/'moments'/name
  args=['python','-u','scripts/locomotion/eval_issue17.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed','1','--checkpoint',model['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--extended_commands','--command_filter','forward_03,yaw_positive,yaw_negative,forward_05,forward_08,stand','--paired_resets','--issue16_mode','original','--issue16_records',str(moments),'--issue17_legacy_clock','--issue17_state_output',str(states),'--output',str(output)]
  with output.with_suffix('.log').open('w') as log:
   child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17states',*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (ROOT/'state_collection_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),child_pid=child.pid,training_seed=seed,completed=len(files),log=str(output.with_suffix('.log')))))
   while child.poll() is None:
    if (ROOT/'stop').exists():os.killpg(child.pid,signal.SIGTERM);child.wait();raise RuntimeError('Stop requested')
    time.sleep(5)
  assert child.returncode==0,seed
  actual=json.loads(output.read_text());original=BASELINE/'clean'/(name+'.json');assert sha(original)==manifest[str(original.relative_to(BASELINE))]
  assert actual['results']==json.loads(original.read_text())['results'] and not actual['issue17_clock']['integer_enabled']
  arrays=0;bindings={str(original):sha(original)}
  for old in (BASELINE/'clean/moments'/name).glob('*.npz'):
   assert sha(old)==manifest[str(old.relative_to(BASELINE))];bindings[str(old)]=sha(old)
   with np.load(old) as a,np.load(moments/old.name) as b:
    for key in a.files:assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes(),(seed,old.name,key)
    arrays+=len(a.files)
  z=np.load(states);assert z['actor'].shape==(20,1024,101) and np.array_equal(z['post_step_indices'],np.arange(0,1000,50))
  files.append(dict(training_seed=seed,path=str(states),sha256=sha(states),count=20480,source_job_sha256=sha(output),original_source_arrays_exact=arrays,original_source_bindings=bindings));print('FIXED_STATE_SOURCE_VERIFIED',seed,arrays,flush=True)
 assert sum(f['count'] for f in files)==102400
 (ROOT/'fixed_state_set.json').write_text(json.dumps(dict(count=102400,files=files,frozen_before_new_model_comparison=True,selection='All1024env at20prespecifiedpoststepindices0,50..950 peroldcontrolseed; no survival/error filtering',source_sha256=sha(__file__)),indent=2)+'\n')
if __name__=='__main__':main()
