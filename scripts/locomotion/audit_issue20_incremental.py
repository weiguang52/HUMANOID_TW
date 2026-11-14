"""Audit every completed phase pair without changing or classifying ongoing evaluations."""
import json,time,numpy as np
from pathlib import Path
from issue20_common import ROOT,sha,save
from issue16_mirror import Mirror101
from issue15_analysis_data import load_yaw
from issue20_analysis import half_cycles,paired

def main():
 records=json.loads((ROOT/'evaluation_records.json').read_text());index={(r['usd'],r['training_seed'],r['eval_seed'],r['phase']):r for r in records};checks=[];arrays=0
 for key,row0 in index.items():
  usd,ts,es,k=key
  if k!=0 or (usd,ts,es,10) not in index:continue
  row1=index[usd,ts,es,10];assert sha(row0['output'])==row0['output_sha256'] and sha(row1['output'])==row1['output_sha256'];d0=json.loads(Path(row0['output']).read_text());d1=json.loads(Path(row1['output']).read_text());a=Path(d0['issue18_tracking']['output']);b=Path(d1['issue18_tracking']['output']);c=json.loads((a/'mirror_runtime_contract.json').read_text());m=Mirror101(c['joint_names'],c['action_names'],c['action_scales']);perm=m.perm.numpy();sign=m.sign.numpy();reset=json.loads((b/'mirrored_reset_audit.json').read_text());assert len(reset)>=4 and all(r['RNG_unchanged'] and max(r['max_errors'].values())==0 for r in reset)
  for r0,r1 in zip(d0['results'],d1['results']):
   cmd=r0['command'];assert cmd==r1['command'];x=np.load(a/(cmd+'_initial_state.npz'));y=np.load(b/(cmd+'_initial_state.npz'));q=x['root'].copy();o=x['env_origins'];q[:,:3]=(q[:,:3]-o)*np.array([1,-1,1],np.float32)+o;q[:,3:7]*=[1,-1,1,-1];q[:,7:10]*=[1,-1,1];q[:,10:13]*=[-1,1,-1]
   errors={'root':float(abs(q-y['root']).max()),'joint_pos':float(abs(x['joint_pos'][:,perm]*sign-y['joint_pos']).max()),'joint_vel':float(abs(x['joint_vel'][:,perm]*sign-y['joint_vel']).max()),'last_action':float(abs(x['last_action'][:,perm]*sign-y['last_action']).max())};assert max(errors.values())==0,(key,cmd,errors)
   _,y0,m0,_=load_yaw(a,cmd,r0);_,y1,m1,_=load_yaw(b,cmd,r1)
   for phase,mom,yaw,masks in [(0,a,y0,m0),(10,b,y1,m1)]:
    idx=np.load(mom/(cmd+'_phase20.npz'))['pre_action_idx'];half_cycles(yaw['heading_yaw'],yaw['heading_start'],masks,idx,phase);steps=json.loads((mom/(cmd+'_step_sha256.json')).read_text());assert len(steps)==1000 and all(len(s)==64 for s in steps);pose=np.load(mom/(cmd+'_replay_env0.npz'))['root_and_joints'];assert pose.shape==(1000,37);quat=pose[:,3:7].astype(float);angle=np.arctan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2));error=float(abs(np.arctan2(np.sin(angle-yaw['heading_yaw'][:,0]),np.cos(angle-yaw['heading_yaw'][:,0]))).max());assert error<=2e-6,error;arrays+=1
   p=paired(y0['heading_yaw'],y0['heading_start'],m0,y1['heading_yaw'],y1['heading_start'],m1);checks.append({'usd':usd,'training_seed':ts,'eval_seed':es,'command':cmd,'initial_state_mirror_errors':errors,'all1000_shifted_phase_steps_verified':True,'paired_trials':1024,'complete20s_pairs':int(p['paired_complete20s'].sum()),'source_pose_yaw_max_error_at_most2e_6':True})
 receipt={'passed':True,'completed_jobs_at_snapshot':len(records),'phase_pair_command_cells':len(checks),'phase_arrays_verified':arrays,'checks':checks,'scientific_classification_performed':False,'timestamp':time.time()};save('incremental_completed_pair_audit.json',receipt);print('INCREMENTAL_COMPLETED_PAIR_AUDIT_PASS',len(records),len(checks),arrays,flush=True)
if __name__=='__main__':main()
