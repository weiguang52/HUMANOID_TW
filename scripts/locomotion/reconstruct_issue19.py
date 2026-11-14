"""Exact source-state reconstruction of new sym_wrap median-heading trial."""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113');REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.chdir(REPO);assert json.loads((ROOT/'evaluation_progress.json').read_text())['state']=='complete';s=json.loads((ROOT/'representative_selection.json').read_text());assert sha(s['checkpoint'])==s['checkpoint_sha256'];folder=ROOT/'reconstruction';folder.mkdir(exist_ok=True);out=folder/'source.json';moments=folder/'moments'
 args=['python','-u','scripts/locomotion/eval_issue19.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed',str(s['eval_seed']),'--checkpoint',s['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--extended_commands','--command_filter','forward_03,yaw_positive,yaw_negative,forward_05,forward_08,stand','--paired_resets','--issue18_mode','original','--issue19_mode','sym_wrap','--issue18_records',str(moments),'--issue18_capture_env',str(s['env_id']),'--issue18_capture_command','forward_05','--output',str(out)]
 assert not out.exists(),'Existing reconstruction requires audit before any retry'
 with (folder/'source.log').open('w') as log:subprocess.run(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17reconstruct',*args],stdout=log,stderr=subprocess.STDOUT,check=True)
 now=json.loads(out.read_text());old=json.loads(Path(s['original_job']).read_text());assert now['results']==old['results'] and now['issue17_clock']['integer_enabled'];oldfolder=Path(old['issue18_tracking']['output']);checks=0
 for before in oldfolder.glob('*.npz'):
  with np.load(before) as a,np.load(moments/before.name) as b:
   for k in a.files:assert a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes(),(before.name,k)
   checks+=len(a.files)
 trajectory=moments/'forward_05_trajectory.npz';meta=trajectory.with_suffix('.json')
 with np.load(trajectory) as z:
  assert z['values'].shape==(1000,43) and z['flags'].shape==(1000,4);flags=z['flags'];q=z['values'][:,3:7].astype(float);yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
 with np.load(oldfolder/'forward_05_yaw.npz') as z:recorded=z['heading_yaw'][:,s['env_id']].astype(float);initial=float(z['heading_start'][s['env_id']])
 assert np.allclose(np.arctan2(np.sin(yaw-recorded),np.cos(yaw-recorded)),0,atol=2e-6,rtol=0)
 data=dict(s,heading_start_yaw=initial,reset_frames=np.flatnonzero(~flags[:,0].astype(bool)).tolist(),trajectory=str(trajectory),trajectory_meta=str(meta),trajectory_metadata=str(meta),trajectory_sha256=sha(trajectory),trajectory_metadata_sha256=sha(meta),all_prior_original_result_rows_exactly_reproduced=True,all_prior_original_recorded_arrays_bitwise_equal=True,compared_arrays=checks,formal_denominators_unchanged=True,integer_clock=True,source_sha256=sha(__file__))
 (ROOT/'reconstruction_state.json').write_text(json.dumps(data,indent=2)+'\n');print('NEW_SYM_INT_RECONSTRUCTION_EXACT',checks,flush=True)
if __name__=='__main__':main()
