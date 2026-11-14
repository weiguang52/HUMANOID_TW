"""Fixed seed1/env0 representatives for both commands, exact saved poses."""
import numpy as np,json,shutil
from issue21_common import ROOT,OLD,REPO,sha,save

def main():
 assert (ROOT/'formal_evaluation_complete').exists();meta=json.loads((ROOT/'evaluation/ground_s1_n16.json').read_text());model=json.loads((ROOT/'cohort.json').read_text())[0];asset=json.loads((OLD/'colored_asset_integrity.json').read_text());assert asset['passed'] and sha(asset['physics_usd'])==asset['physics_sha256'] and sha(asset['colored_usd'])==asset['colored_sha256']
 new=ROOT/'display_symmetric_colored.usd';shutil.copy2(asset['colored_usd'],new);assert sha(new)==asset['colored_sha256'];asset['colored_usd']=str(new);save('colored_asset_integrity.json',asset);shutil.copy2(OLD/'joint_colors.json',ROOT/'joint_colors.json')
 plan=[]
 for command in ['stand','forward_05']:
  values={}
  for k in [0,10]:
   p=ROOT/'evaluation'/('ground_s1_n16_%s_closed_k%d.npz'%(command,k));z=np.load(p);poses=np.concatenate([z['root'][:,0,:7],z['joint_pos'][:,0]],axis=1);q=z['root'][:,0,3:7];yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2));qi=z['initial_root'][0,3:7];start=np.arctan2(2*(qi[0]*qi[3]+qi[1]*qi[2]),1-2*(qi[2]**2+qi[3]**2));failed=np.flatnonzero(z['terminated'][:,0]);traj=ROOT/('%s_k%d_replay.npz'%(command,k));yp=ROOT/('%s_k%d_yaw.npz'%(command,k));np.savez_compressed(traj,root_and_joints=poses);np.savez_compressed(yp,heading_yaw=yaw[:,None],heading_start=np.array([start]),first_failure_step=np.array([int(failed[0]) if len(failed) else -1]));values[k]=(p,traj,yp,poses)
  xy=np.concatenate([x[3][:,:2] for x in values.values()]);center=(xy.min(0)+xy.max(0))/2;span=max(float(np.ptp(xy,axis=0).max())*1.15,1.)
  for k,(source,traj,yp,poses) in values.items():
   s={'phase':k,'usd':'symmetric','training_seed':1,'eval_seed':1,'env_id':0,'num_envs':16,'command':command,'checkpoint':model['checkpoint'],'checkpoint_sha256':model['sha256'],'trajectory':str(traj),'trajectory_sha256':sha(traj),'yaw_file':str(yp),'yaw_sha256':sha(yp),'source_trace':str(source),'source_trace_sha256':sha(source),'joint_names':meta['joint_names'],'common_xy_center':center.tolist(),'common_xy_span':span,'selection_rule':'fixed training seed1/eval1/env0 from num_envs16, declared before full cohort results; no worst-trial selection'};save('replay_%s_k%d.json'%(command,k),s);plan.append(s)
 save('media_selection.json',plan);print('ISSUE21_MEDIA_SELECTED',len(plan),flush=True)
if __name__=='__main__':main()
