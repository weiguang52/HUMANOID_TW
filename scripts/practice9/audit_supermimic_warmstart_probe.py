"""Compare cache intervention with native replay noise and frozen-model error."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
p=argparse.ArgumentParser(description=__doc__)
for key in ['collection','model','source-checkpoint']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();root=a.collection;assert not (root/'audit.json').exists();report=json.loads((root/'results.json').read_text());plan=report['plan'];rows=report['rows'];assert len(rows)==30
assert hashlib.sha256(a.source_checkpoint.read_bytes()).hexdigest()==plan['source_checkpoint_sha256']
source=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False);c=torch.load(a.model,map_location='cpu',weights_only=False)
for key in ['tw_state_joint_names','tw_action_joint_names','tw_physics_contract']:assert source['cfg']['extra'][key]==c['cfg']['extra'][key]
assert plan['physics_contract']==c['cfg']['extra']['tw_physics_contract']
expected={(profile,w,eps,condition) for profile,w in plan['profiles'] for eps in plan['epsilons'] for condition in plan['conditions']};table={(r['profile'],r['warmup'],r['epsilon'],r['condition']):r for r in rows};assert table.keys()==expected
loaded={};files=[]
for key,row in table.items():
 path=(root/row['file']).resolve();assert path.is_relative_to(root.resolve())
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 for name,width in [('state',73),('context',34),('action',30),('next_state',73),('warmstart_before',36),('warmstart_after',36)]:assert d[name].shape==(61,width) and np.isfinite(d[name]).all()
 assert d['terminated'].shape==(61,) and int(d['terminated'].sum())==row['terminated_count']
 assert row['visible_state_unchanged_by_intervention'] and row['initial_state_error']<=plan['replay_atol']
 action=np.zeros((61,30),np.float32);action[np.arange(1,61),np.repeat(np.arange(30),2)]=np.tile([-1,1],30)*row['epsilon'];np.testing.assert_array_equal(action,d['action'])
 if row['condition']=='zero_warmstart':assert not np.any(d['warmstart_after'])
 else:np.testing.assert_array_equal(d['warmstart_before'],d['warmstart_after'])
 loaded[key]=d;files.append(dict(file=row['file'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
cfg=c['cfg'];w=c['world_model'];base=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();base.load_state_dict(w);base.eval();model=apply_offline_prediction_contract(base,c)
def metrics(x):return dict(rmse=float(np.sqrt(np.mean(x.astype(np.float64)**2))),abs_max=float(np.max(np.abs(x))))
cases=[];pooled={name:[] for name in ['cache_effect','native_repeat','actual_step_change','model_error']}
for profile,warmup in plan['profiles']:
 for eps in plan['epsilons']:
  native,zero,repeat=[loaded[(profile,warmup,eps,condition)] for condition in plan['conditions']]
  for other in [zero,repeat]:np.testing.assert_allclose(other['state'],native['state'],rtol=0,atol=plan['replay_atol']);np.testing.assert_allclose(other['context'],native['context'],rtol=0,atol=1e-7);np.testing.assert_array_equal(other['action'],native['action'])
  with torch.no_grad():pred=model(*[torch.tensor(native[k],device='cuda:0') for k in ['state','action','context']]).cpu().numpy()
  assert np.isfinite(pred).all();difference={'cache_effect':zero['next_state']-native['next_state'],'native_repeat':repeat['next_state']-native['next_state'],'actual_step_change':native['next_state']-native['state'],'model_error':pred-native['next_state']};groups={}
  for group,sl in [('root_position',slice(0,3)),('root_linear_velocity',slice(7,10)),('root_angular_velocity',slice(10,13)),('joint_position',slice(13,43)),('joint_velocity',slice(43,73))]:groups[group]={k:metrics(v[:,sl]) for k,v in difference.items()}
  for key,v in difference.items():pooled[key].append(v[:,10:13])
  cases.append(dict(profile=profile,warmup=warmup,epsilon=eps,states=61,warmstart_nonzero=bool(np.any(native['warmstart_before'])),initial_zero_state_error=float(np.max(np.abs(zero['state']-native['state']))),terminated_counts=[int(d['terminated'].sum()) for d in [native,zero,repeat]],groups=groups))
summary={key:metrics(np.concatenate(values)) for key,values in pooled.items()}
out=dict(scope='Initial solver-cache intervention in isolated diagnostic instances, all planned cases. No production physics changes. Visible state replay tolerance1e-5 retained; output differences compared with repeat noise. This tests qacc_warmstart only, not every hidden solver buffer or universal Markovity.',physics_contract=plan['physics_contract'],solver_warmstart_disabled=plan['warmstart_disabled'],model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),files=files,cases=cases,pooled_root_angular_velocity=summary,promoted=False)
nonzero=[r for r in cases if r['warmstart_nonzero']]
assert nonzero
nonzero_summary=dict(cases=len(nonzero),root_angular_velocity_rmse={k:float(np.sqrt(np.mean([r['groups']['root_angular_velocity'][k]['rmse']**2 for r in nonzero]))) for k in pooled},scope='Equal61states per condition, nonzero native-cache cases only')
(root/'nonzero_cache_summary.json').write_text(json.dumps(nonzero_summary,indent=2,allow_nan=False))
(root/'audit.json').write_text(json.dumps(out,indent=2,allow_nan=False));(root/'status').write_text('audited_probe_complete\n');print(json.dumps(dict(solver_warmstart_disabled=out['solver_warmstart_disabled'],cases=len(cases),summary=summary)))
