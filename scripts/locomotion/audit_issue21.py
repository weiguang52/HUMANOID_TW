"""All-job input/protocol and independent trace consistency audit."""
import json,numpy as np,subprocess
from pathlib import Path
from issue21_common import ROOT,OLD,REPO,sha,save
from issue21_analysis import Mirror,slope

def main():
 assert (ROOT/'formal_evaluation_complete').exists();jobs=json.loads((ROOT/'evaluation_records.json').read_text());assert len(jobs)==25;models=json.loads((ROOT/'cohort.json').read_text());cold=json.loads((ROOT/'cold_start_records.json').read_text());assert len(cold)==155
 for rec in cold:
  assert sha(rec['output'])==rec['sha256'];sub=json.loads(Path(rec['output']).read_text());assert sub['cold_start'] and sub['model_unchanged'] and len(sub['records'])==1
 for ts in range(1,6):
  control=json.loads((ROOT/'controls'/('s%d.json'%ts)).read_text());assert control['cold_start'] and control['records'][0]['epsilon']==0
  for command in ['stand','forward_05']:
   for kind,mode in [('closed','ground'),('open','ground'),('open','air')]:
    a=np.load(ROOT/'evaluation'/('%s_s%d_n16_%s_%s_k0.npz'%(mode,ts,command,kind)));b=np.load(ROOT/'evaluation'/('%s_s%d_n1_%s_%s_k0.npz'%(mode,ts,command,kind)))
    for key in ['root','joint_pos','joint_vel']:np.testing.assert_array_equal(a['initial_'+key][0],b['initial_'+key][0])
 expected={(mode,ts,n) for mode,ns in [('ground',[16,1]),('air',[16,1]),('perturb',[1024])] for ts in range(1,6) for n in ns};seen=set();traces=0;perturb=0;initial_errors=[]
 for p,digest in json.loads((ROOT/'frozen_evaluation_sources.json').read_text())['files'].items():assert sha(p)==digest,p
 assert sha(ROOT/'classification_rules.json')=='5d017196e31b317135bbceda1fa5ebd46f7ec82b662030432c434044987c4192'
 for job in jobs:
  p=Path(job['output']);assert sha(p)==job['sha256'];meta=json.loads(p.read_text());mode=meta['mode'];ts=meta['training_seed'];n=meta['num_envs'];seen.add((mode,ts,n));assert meta['checkpoint_sha256']==models[ts-1]['sha256']==sha(meta['checkpoint']);assert meta['all_fresh_processes'] and meta['model_unchanged'] and meta['clean'] and meta['training_updates']==0 and meta['step_dt']==.02;assert meta['strict_usd_sha256']==sha(OLD/'strict_symmetric.usd');m=Mirror(meta['joint_names'],meta['contact_body_names'])
  if mode=='perturb':
   assert len(meta['records'])==6 and {x['epsilon'] for x in meta['records']}=={-.2,-.05,-.01,.01,.05,.2}
  else:assert len(meta['records'])==(8 if mode=='ground' else 4)
  for rec in meta['records']:
   p=Path(rec['path']);assert sha(p)==rec['sha256'];z=np.load(p)
   for key in z.files:assert np.isfinite(z[key]).all(),(p,key)
   if mode=='perturb':
    assert z['yaw'].shape==(1000,1024) and z['terminated'].shape==(1000,1024);values,valid=slope(z['yaw'],z['terminated']);t=np.arange(1,1001)*.02;pick=(t>=5)&(t<=20);design=np.stack([np.ones(pick.sum()),t[pick]],axis=1);ind=np.linalg.lstsq(design,np.unwrap(z['yaw'].astype(float),axis=0)[pick],rcond=None)[0][1]*180/np.pi;np.testing.assert_allclose(values[valid],ind[valid],rtol=1e-11,atol=1e-11);perturb+=1;continue
   steps=rec['steps'];assert z['obs'].shape==(steps,n,101) and z['action'].shape==(steps,n,30) and z['root'].shape==(steps,n,13) and z['contact'].shape==(steps,n,len(meta['contact_body_names']),3);assert z['terminated'].shape==(steps,n)
   for key in ['applied_torque','estimated_pd_torque','explicit_actuation_force','joint_target']:assert z[key].shape==(steps,n,30),key
   if mode=='air':assert np.max(np.linalg.norm(z['contact'],axis=-1))==0,'Air fixture made contact'
   if rec['kind']=='open':
    src=ROOT/'evaluation'/('ground_s%d_n%d_%s_closed_k0.npz'%(ts,n,rec['command']));actions=np.load(src)['action'][:200];expected_actions=actions if rec['phase']==0 else m.joint(actions);np.testing.assert_array_equal(z['action'],expected_actions)
   if rec['phase']==0:
    other=np.load(p.with_name(p.name.replace('_k0.npz','_k10.npz')))
    np.testing.assert_array_equal(other['initial_joint_pos'],m.joint(z['initial_joint_pos']));np.testing.assert_array_equal(other['initial_joint_vel'],m.joint(z['initial_joint_vel']));e=np.abs(other['initial_root']-m.root(z['initial_root'])).max();assert e<=1e-6;e=float(e);initial_errors.append(e)
   traces+=1
 assert seen==expected and traces==120 and perturb==30
 assert not subprocess.check_output(['git','diff','--name-only','b8ddc926','--','source','scripts/locomotion/eval_cmd.py','scripts/locomotion/train.py'])
 save('independent_protocol_audit.json',{'passed':True,'jobs':len(jobs),'cold_processes':155,'matched_zero_epsilon_controls':5,'single_env_initial_state_exact_env0_of16':True,'trace_files':traces,'perturbation_cells':perturb,'air_contacts_zero':True,'all_open_loop_actions_exact_frozen_ground_k0_or_mirror':True,'mirrored_initial_joint_states_exact':True,'max_initial_root_origin_precision_error':max(initial_errors),'all_raw_finite':True,'all_perturbation_slopes_independent_lstsq_match':True,'all_frozen_sources_unchanged':True,'all_model_states_unchanged':True,'training_updates':0,'defaults_unchanged':True});print('ISSUE21_INDEPENDENT_PROTOCOL_AUDIT_PASS',traces,perturb,flush=True)
if __name__=='__main__':main()
