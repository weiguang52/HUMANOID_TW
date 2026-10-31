"""Independent replay archive checks and same-state model/native error comparison."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_geometry_model import configure_offline_geometry
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
p=argparse.ArgumentParser(description=__doc__)
for key in ['replay-root','collection','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--checkpoint',type=Path,action='append',required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
source_rows=json.loads((a.collection/'results.json').read_text())['rows'];assert len(source_rows)==16
phase_reports={};cold_pools={};evidence=[]
def metrics(pred,target):
 e=pred-target
 return {name:float(np.sqrt(np.mean(e[:,sl]**2))) for name,sl in [('root_position_rmse_m',slice(0,3)),('root_linear_velocity_rmse_m_s',slice(7,10)),('root_angular_velocity_rmse_rad_s',slice(10,13)),('joint_position_rmse_rad',slice(13,43)),('joint_velocity_rmse_rad_s',slice(43,73))]}
for phase in ['warm','cold']:
 root=a.replay_root/phase;plan=json.loads((root/'plan.json').read_text());result=json.loads((root/'results.json').read_text());assert not plan['smoke'] and not result['smoke']
 assert len(result['rows'])==16 and [r['file'] for r in result['rows']]==[r['trace'] for r in source_rows]
 hashes={r['file']:r['sha256'] for r in plan['source_files']};rows=[];total=0
 for original,row in zip(source_rows,result['rows']):
  source=a.collection/original['trace'];assert hashlib.sha256(source.read_bytes()).hexdigest()==hashes[original['trace']]
  with np.load(source,allow_pickle=False) as z:stored={k:z[k].copy() for k in ['state','action','context','next_state']}
  path=root/row['trace'];assert path.resolve().is_relative_to(root.resolve())
  with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  assert all(np.isfinite(v).all() for v in d.values())
  n=len(stored['state']);idx=d['index'].astype(int)
  expected=np.arange(n) if phase=='warm' else np.array(sorted(set(range(min(20,n)))|set(range(0,n,50))|{n-1}))
  np.testing.assert_array_equal(idx,expected);np.testing.assert_array_equal(d['recorded_next_state'],stored['next_state'][idx])
  np.testing.assert_allclose(d['context'],stored['context'][idx],rtol=0,atol=plan['context_atol'])
  actual=d['actual_next_state'];target=d['recorded_next_state'];initial_error=np.max(np.abs(d['restored_state']-stored['state'][idx]),axis=1)
  if phase=='warm':np.testing.assert_array_equal(actual[:-1],d['restored_state'][1:])
  else:assert np.max(initial_error)<=plan['state_write_atol']
  error=np.max(np.abs(actual-target),axis=1);bad=idx[error>plan['native_replay_atol']]
  rows.append(dict(file=original['trace'],frames=len(idx),max_input_error=float(initial_error.max()),max_next_state_error=float(error.max()),native_precision_failure_count=len(bad),native_precision_failure_indices=bad.tolist(),metrics=metrics(actual,target),terminated_indices=idx[d['terminated'].astype(bool)].tolist(),original_termination_reasons=original['termination_reasons']))
  evidence.append(dict(phase=phase,file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  total+=len(idx)
  if phase=='cold':
   pool=cold_pools.setdefault(original['split'],{k:[] for k in ['state','action','context','native','recorded']})
   for k,v in [('state',d['restored_state']),('action',stored['action'][idx]),('context',d['context']),('native',actual),('recorded',target)]:pool[k].append(v)
 phase_reports[phase]=dict(frames=total,all_within_original_native_atol=all(x['native_precision_failure_count']==0 for x in rows),failed_frames=sum(x['native_precision_failure_count'] for x in rows),max_next_state_error=max(x['max_next_state_error'] for x in rows),rows=rows)
assert phase_reports['warm']['frames']==7775
cold_pools={split:{k:np.concatenate(v) for k,v in pool.items()} for split,pool in cold_pools.items()}
model_reports=[]
for path in a.checkpoint:
 c=torch.load(path,map_location='cpu',weights_only=False);w=c['world_model'];cfg=c['cfg']
 mapping=json.loads((a.collection/'control_mapping.json').read_text());assert mapping['state_joint_names']==cfg['extra']['tw_state_joint_names'] and mapping['action_joint_names']==cfg['extra']['tw_action_joint_names']
 assert cfg['extra']['tw_mjcf_sha256']==plan['asset_sha256']
 base=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda()
 configure_offline_geometry(base,c);base.load_state_dict(w);base.eval();model=apply_offline_prediction_contract(base,c);rows=[]
 with torch.no_grad():
  for split,d in cold_pools.items():
   predictions=[]
   for start in range(0,len(d['state']),128):predictions.append(model(*[torch.tensor(d[k][start:start+128],device='cuda:0') for k in ['state','action','context']]).cpu().numpy())
   pred=np.concatenate(predictions);assert np.isfinite(pred).all()
   rows.append(dict(split=split,states=len(pred),native_vs_recorded=metrics(d['native'],d['recorded']),model_vs_native=metrics(pred,d['native']),model_vs_recorded=metrics(pred,d['recorded'])))
  for k,v in base.state_dict().items():assert torch.equal(v.cpu(),w[k]),k
 model_reports.append(dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,weights_unchanged=True))
print(json.dumps(dict(phases={k:{field:v[field] for field in ['frames','all_within_original_native_atol','failed_frames','max_next_state_error']} for k,v in phase_reports.items()},model_angular=[dict(sha256=m['sha256'],rows=[dict(split=x['split'],native_vs_recorded=x['native_vs_recorded']['root_angular_velocity_rmse_rad_s'],model_vs_native=x['model_vs_native']['root_angular_velocity_rmse_rad_s']) for x in m['rows']]) for m in model_reports])),flush=True)
out=dict(phases=phase_reports,models=model_reports,evidence=evidence,archive_invariants_passed=True,scope='All7775 warm transitions and fixed cold subset only. Warm drift includes changed inputs; cold precision failures retained at original1e-5. Model/native compared at identical restored states and saved actions/context; no fitting, threshold change, universal Markov proof or control/policy acceptance.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
