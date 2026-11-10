"""Issue12 matched groups; issue11 bins, full denominators and raw checks retained."""
import json,hashlib
from pathlib import Path
import numpy as np
from risk_analysis_utils import quartile,directions,phase_bin,push_windows,AMP_CUTS,MASS_CUTS
ROOT=Path('/root/gpufree-data/tw_stage1/issue12_20261110')
COMMANDS=['forward_03','forward_05','forward_08']

def load_dataset():
 assert (ROOT/'evaluation_complete').exists();models=json.loads((ROOT/'final_models.json').read_text());jobs=[];mu=[];pairing=[];reference={}
 for group in ['control','treatment']:
  for seed in range(1,6):
   name=f'{group}_s{seed}'
   assert hashlib.sha256(Path(models[name]['checkpoint']).read_bytes()).hexdigest()==models[name]['sha256']
   for es in [1,2,3]:
    output=json.loads((ROOT/f'{name}_eval{es}.json').read_text());assert output['policy_last_action_params']=={} and output['critic_last_action_params']=={} and output['paired_resets'] and output['deterministic_policy']
    cfg=output['perturbation_config'];assert cfg['mass_distribution_params']==[.5,1.5] and cfg['push_velocity_range']=={'x':[-.3,.3],'y':[-.3,.3]}
    assert [x['command'] for x in output['results']]==COMMANDS
    for command in COMMANDS:
     folder=ROOT/'traces'/f'{name}_eval{es}';meta=json.loads((folder/f'{command}_trials.json').read_text());raw=dict(np.load(folder/f'{command}_trials.npz'))
     assert meta['all_trials']==1024 and len(raw['env_ids'])==1024 and meta['step_dt']==.02 and meta['torch_rng_invariance_checks']>=2
     assert meta['observer_no_sampling_or_physics_writes'] and all(np.isfinite(v).all() for v in raw.values())
     assert np.array_equal(raw['env_ids'],np.arange(1024)) and np.array_equal(raw['failed'],raw['first_failure_step']>=0)
     result=next(x for x in output['results'] if x['command']==command);assert int(raw['failed'].sum())==result['failed_trials']
     assert (raw['masses']>0).all() and np.allclose(raw['mass_ratios'],raw['masses']/raw['nominal_masses'])
     base_id=meta['body_names'].index('base_link');assert ((raw['mass_ratios'][:,base_id]>=.5-1e-6)&(raw['mass_ratios'][:,base_id]<=1.5+1e-6)).all()
     assert ((raw['materials'][:,:,0]>=.4-1e-6)&(raw['materials'][:,:,0]<=1.4+1e-6)).all() and ((raw['materials'][:,:,1]>=.3-1e-6)&(raw['materials'][:,:,1]<=1.2+1e-6)).all()
     if es not in reference:reference[es]={k:raw[k].copy() for k in ['masses','materials']}
     pairing.append({'model':name,'eval_seed':es,'command':command,'masses_equal_eval_reference':bool(np.array_equal(reference[es]['masses'],raw['masses'])),'materials_equal_eval_reference':bool(np.array_equal(reference[es]['materials'],raw['materials']))})
     mu.extend((es,int(i),float(s),float(d)) for i,s,d in zip(raw['env_ids'],raw['foot_static_min'],raw['foot_dynamic_min']))
     jobs.append((group,seed,es,name,command,folder,meta,raw))
 cuts=json.loads((ROOT/'frozen_feature_cuts.json').read_text())
 provenance=json.loads((ROOT/'feature_cuts_provenance.json').read_text())
 assert hashlib.sha256((ROOT/'frozen_feature_cuts.json').read_bytes()).hexdigest()==provenance['source_sha256']
 assert np.array_equal(np.asarray(cuts['push_amplitude']),AMP_CUTS)
 assert np.array_equal(np.asarray(cuts['mass']),MASS_CUTS)
 assert all(x['masses_equal_eval_reference'] and x['materials_equal_eval_reference'] for x in pairing)
 previous=Path('validation_artifacts/stage1_issue11_20261110/traces')
 for es,values in reference.items():
  z=np.load(previous/f'requested_s1_u4500_eval{es}'/'forward_03_trials.npz')
  assert np.array_equal(values['masses'],z['masses']) and np.array_equal(values['materials'],z['materials'])
 (ROOT/'randomization_pairing_audit.json').write_text(json.dumps({'passed':True,'rows':pairing,'all_same_mass_and_material_draws':True,'exactly_match_issue11_startup_draws':True,'shared_physical_streams':3072},indent=2))
 trials=[];pushes=[];failure_cases=[];audits=[]
 for group,seed,es,name,command,folder,meta,z in jobs:
  body=meta['body_names'];bi=body.index('base_link');li=body.index('left_foot');ri=body.index('right_foot');fields=meta['push_fields'];p=z['pushes']
  def col(n):return p[:,slice(*fields[n])]
  ids=col('env_id')[:,0].astype(int);steps=col('push_step')[:,0].astype(int);at_risk=col('first_trial_at_risk')[:,0].astype(bool)
  eligible,event,censored=push_windows(ids,steps,at_risk,z['first_failure_step'])
  assert not censored.any(),'Unexpected overlapping at-risk push windows: inspect censoring before analysis'
  dw=col('delta_world');db=col('delta_body');q=col('root_quaternion');t=2*np.cross(q[:,1:],dw);expected=db*0+dw-q[:,:1]*t+np.cross(q[:,1:],t)
  assert np.allclose(db,expected,atol=2e-6,rtol=2e-6) and np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-5)
  assert np.max(np.abs(dw[:,:2]))<=.30001 and np.max(np.abs(dw[:,2]))<=1e-5
  assert np.allclose(col('root_velocity_after')-col('root_velocity_before'),dw,atol=1e-6)
  contact=col('contact');support=col('support_code')[:,0].astype(int);assert np.array_equal(support,contact[:,0]+2*contact[:,1]);assert np.isin(support,[0,1,2,3]).all()
  phase=col('cycle_phase');assert ((phase>=-1)&(phase<=1)).all();phase_codes=phase_bin(phase,support)
  amp=np.linalg.norm(db,axis=1);ampq=quartile(amp,AMP_CUTS);direction=directions(db);delay=col('delay20ms')[:,0].astype(int);assert np.isin(delay,[0,1]).all();assert np.array_equal(delay[at_risk],z['initial_delay20ms'][ids[at_risk]].astype(int))
  massq=quartile(z['mass_ratios'][:,bi],MASS_CUTS);staticq=quartile(z['foot_static_min'],cuts['static']);dynamicq=quartile(z['foot_dynamic_min'],cuts['dynamic'])
  single=np.isin(support,[1,2]);strict=(staticq[ids]==1)&(massq[ids]==4)&(delay==1)&(db[:,0]>=.3)&single
  envelope=((staticq[ids]==1)|(dynamicq[ids]==1))&np.isin(massq[ids],[1,4])&(delay==1)&(ampq==4)&single
  trial_lookup={}
  for i in range(1024):
   tr={'trial_id':f'{group}_s{seed}_e{es}_{command}_n{i}','group':group,'training_seed':seed,'eval_seed':es,'model':name,'command':command,'env_id':i,'failed':int(z['failed'][i]),'first_failure_step':int(z['first_failure_step'][i]),'base_mass_ratio':float(z['mass_ratios'][i,bi]),'left_foot_mass_ratio':float(z['mass_ratios'][i,li]),'right_foot_mass_ratio':float(z['mass_ratios'][i,ri]),'foot_static_min':float(z['foot_static_min'][i]),'foot_dynamic_min':float(z['foot_dynamic_min'][i]),'delay_ms':20*int(z['initial_delay20ms'][i]),'mass_q':int(massq[i]),'static_q':int(staticq[i]),'dynamic_q':int(dynamicq[i]),'strict_exposed':False,'envelope_exposed':False,'strict_proximal_failure':False,'envelope_proximal_failure':False}
   trial_lookup[i]=tr;trials.append(tr)
  perfoot={}
  for side in ['left_foot','right_foot']:
   ss=meta['body_shape_indices'][side];assert ss;perfoot[side]=(z['materials'][:,ss,0].min(1),z['materials'][:,ss,1].min(1))
  for j,env_id in enumerate(ids):
   tr=trial_lookup[int(env_id)];feet=col('foot_positions_local')[j].reshape(2,3);com=col('whole_body_com_local')[j];vbefore=col('whole_body_com_velocity_before')[j];vafter_proxy=vbefore+dw[j];h=float(com[2]);cp=com[:2]+vafter_proxy[:2]/np.sqrt(9.81/max(h,.01))
   active=np.flatnonzero(contact[j]);distance=-1.;mu_support=-1.;stopping=-1.
   if len(active):
    points=feet[active,:2]
    if len(active)==1:nearest=points[0]
    else:
     segment=points[1]-points[0];alpha=np.clip(np.dot(cp-points[0],segment)/max(float(np.dot(segment,segment)),1e-12),0,1);nearest=points[0]+alpha*segment
    distance=float(np.linalg.norm(cp-nearest));mu_support=min(float(perfoot[['left_foot','right_foot'][k]][1][env_id]) for k in active);stopping=float(np.dot(vafter_proxy[:2],vafter_proxy[:2])/(2*9.81*mu_support))
   row={'trial_id':tr['trial_id'],'group':group,'training_seed':seed,'eval_seed':es,'model':name,'command':command,'env_id':int(env_id),'push_step':int(steps[j]),'push_second':float(steps[j]*.02),'at_risk':bool(at_risk[j]),'eligible':bool(eligible[j]),'failure_within1s':int(event[j]),'mass_q':int(massq[env_id]),'static_q':int(staticq[env_id]),'dynamic_q':int(dynamicq[env_id]),'delay_ms':20*int(delay[j]),'amplitude':float(amp[j]),'amplitude_q':int(ampq[j]),'direction':str(direction[j]),'side_sign':('left' if db[j,1]>=0 else 'right') if direction[j]=='side' else str(direction[j]),'support_code':int(support[j]),'phase_bin':int(phase_codes[j]),'left_cycle_phase':float(phase[j,0]),'right_cycle_phase':float(phase[j,1]),'delta_body_x':float(db[j,0]),'delta_body_y':float(db[j,1]),'delta_body_z':float(db[j,2]),'delta_world_x':float(dw[j,0]),'delta_world_y':float(dw[j,1]),'com_height':h,'com_vx_before':float(vbefore[0]),'com_vy_before':float(vbefore[1]),'capture_distance_to_support_centers_proxy':distance,'support_dynamic_min':mu_support,'friction_stopping_distance_proxy':stopping,'strict_corner':bool(strict[j]),'envelope_corner':bool(envelope[j]),'source_record':str(folder/f'{command}_trials.npz'),'source_push_row':j}
   pushes.append(row)
   if eligible[j]:
    for corner,flag in [('strict',strict[j]),('envelope',envelope[j])]:
     if flag:tr[corner+'_exposed']=True
     if flag and event[j]:tr[corner+'_proximal_failure']=True
  fm=json.loads((folder/f'{command}.json').read_text());fz=np.load(folder/f'{command}.npz');assert len(fm['records'])==int(z['failed'].sum()) and np.array_equal(fz['failed'],z['failed'])
  for index,rec in enumerate(fm['records']):
   i=rec['env_id'];candidates=np.flatnonzero((ids==i)&at_risk&(steps<rec['failure_step']));last=int(candidates[np.argmax(steps[candidates])]) if len(candidates) else None
   recent=last is not None and rec['failure_step']-steps[last]<=50
   mode=f"delay{20*int(delay[last])}__{direction[last]}__support{int(support[last])}" if recent else 'no_recent_push'
   vv=fz['values'][index][fz['valid'][index]];ff=fm['fields'];tilt=vv[-1,slice(*ff['roll'])][0];pitch=vv[-1,slice(*ff['pitch'])][0]
   assert np.isfinite(vv).all() and np.allclose(np.linalg.norm(vv[:,slice(*ff['root_pose_local'])][:,3:],axis=1),1,atol=1e-5)
   if 'base_height' in rec['termination_terms']:assert vv[-1,slice(*ff['base_height'])][0]<.080001
   if last is not None:
    assert np.allclose(vv[-1,slice(*ff['last_push_delta_world'])],dw[last],atol=1e-6)
    assert np.allclose(vv[-1,slice(*ff['last_push_delta_body'])],db[last],atol=1e-6)
    assert abs(float(vv[-1,slice(*ff['seconds_since_push'])][0])-(rec['failure_step']-steps[last])*.02)<1e-5
   failure_cases.append({'trial_id':trial_lookup[i]['trial_id'],'group':group,'training_seed':seed,'eval_seed':es,'model':name,'command':command,'env_id':i,'failure_step':rec['failure_step'],'mode':mode,'last_push_row':last,'recent_push':bool(recent),'last_push_amplitude':None if last is None else float(amp[last]),'termination_terms':rec['termination_terms'],'terminal_roll':float(tilt),'terminal_pitch':float(pitch),'terminal_near_soft_limit_count':int(vv[-1,slice(*ff['joint_near_soft_limit'])].sum()),'terminal_near_hard_limit_count':int(vv[-1,slice(*ff['joint_near_hard_limit'])].sum()),'source_npz':str(folder/f'{command}.npz'),'source_metadata':str(folder/f'{command}.json'),'record_index':index})
  audits.append({'model':name,'eval_seed':es,'command':command,'trials':1024,'failed_trials':int(z['failed'].sum()),'all_pushes':len(p),'at_risk_pushes':int(at_risk.sum()),'eligible1s_pushes':int(eligible.sum()),'administrative_tail_excluded':int((at_risk&~eligible).sum()),'overlap_censored':int(censored.sum()),'observer_rng_checks':meta['torch_rng_invariance_checks']})
 assert len(trials)==92160 and len(audits)==90
 (ROOT/'raw_record_integrity_audit.json').write_text(json.dumps({'passed':True,'command_records':audits,'all_trials':len(trials),'all_pushes':len(pushes),'all_first_failures':len(failure_cases),'coordinate_delta_norm_and_delay_support_checks':True},indent=2))
 return trials,pushes,failure_cases,cuts
