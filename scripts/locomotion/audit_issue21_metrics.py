"""Independently verify all component crossing CSV entries against signed arrays."""
import json,csv,numpy as np
from issue21_common import ROOT,save,sha
from issue21_analysis import GROUPS

def main():
 pairs=json.loads((ROOT/'pair_analysis.json').read_text());bycase={x['case']:x for x in pairs};cache={};count=0;last_key=None;last_flat=None
 for x in pairs:
  meta=json.loads((ROOT/'evaluation'/('%s_s%d_n%d.json'%(x['mode'],x['training_seed'],x['num_envs']))).read_text());cache[x['case']]={'z':np.load(x['residual_file']),'joint_names':meta['joint_names'],'body_names':meta['contact_body_names']}
 with (ROOT/'first_component_crossings.csv').open() as f:
  for row in csv.DictReader(f):
   data=cache[row['case']];name=row['quantity'];names=data['joint_names'] if name in ['joint_pos','joint_vel','applied_torque','action','joint_pos_obs','joint_vel_obs','last_action','estimated_pd_torque','explicit_actuation_force','joint_target'] else [n+'.'+axis for n in data['body_names'] for axis in ['Fx','Fy','Fz']] if name=='contact' else {'ang_vel':['wx','wy','wz'],'gravity':['gx','gy','gz'],'commands':['vx','vy','wz'],'gait_phase':['sin','cos'],'root_position':['x','y','z'],'root_quaternion':['qw','qx','qy','qz'],'root_linear_velocity':['vx','vy','vz'],'root_angular_velocity':['wx','wy','wz']}[name];ci=names.index(row['component']);key=(row['case'],name)
   if key!=last_key:
    e=data['z'][name+'_signed'];last_flat=e.reshape(e.shape[0],e.shape[1],-1);last_key=key
   v=last_flat[:,:,ci];hits=np.nonzero(abs(v)>float(row['threshold']))
   if len(hits[0]):
    t=int(hits[0][0]);env=int(hits[1][0]);offset=0 if name in GROUPS or name=='action' else 1;assert int(row['step'])==t+offset and int(row['env'])==env;assert float(row['signed_residual'])==float(v[t,env])
   else:assert row['step']==row['env']==row['signed_residual']==''
   count+=1
 assert len(pairs)==60 and count>60000
 save('independent_component_crossing_audit.json',{'passed':True,'audit_source_sha256':sha(__file__),'pairs':len(pairs),'component_threshold_rows':count,'all_signed_component_crossings_recomputed':True,'all_no_crossing_entries_verified':True,'physical_post_step_offset_and_observation_pre_step_offset_verified':True});print('ISSUE21_ALL_COMPONENT_CROSSINGS_AUDIT_PASS',count,flush=True)
if __name__=='__main__':main()
