"""Independent finite-probe budget, sample, frozen-source and control audit."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['smoke4','fit'],required=True);p.add_argument('--reference-control',type=Path,required=True);a=p.parse_args();updates=4 if a.stage=='smoke4' else 1000
rows=[];histories={};artifacts={}
for arm in ['control','candidate']:
 root=a.root/arm/a.stage;path=root/'world_model_calibrated.pt';x=torch.load(path,map_location='cpu',weights_only=False);plan=x['calibration_plan'];artifacts[arm]=x
 assert x['calibration_updates']==updates and plan['dual_loss_step']==(arm=='candidate')
 assert plan['actual_prediction_points_min']==plan['actual_prediction_points_max']==14816
 assert plan['actual_prediction_points_total']==14816*updates
 assert len(x['optimizer']['state'])==8 and all(int(s['step'])==updates for s in x['optimizer']['state'].values())
 h=json.loads((root/'dual_loss_history.json').read_text());assert len(h)==updates;histories[arm]=h
 for i,v in enumerate(h,1):
  assert v['update']==i and v['enabled']==(arm=='candidate') and len(v['proposals'])==12
  expected=[(name,factor) for name in ['full_saved_momentum','full_zero_first_moment','static_saved_momentum','static_zero_first_moment'] for factor in [1.,.5,.25]]
  assert [(z['direction'],z['factor']) for z in v['proposals']]==expected
  for z in v['proposals']:assert z['accepted']==(z['static_loss'] is not None and z['full_loss'] is not None and z['static_loss']<=v['baseline_static'] and z['full_loss']<=v['baseline_full'])
  if arm=='candidate':
   first=next((z for z in v['proposals'] if z['accepted']),None)
   assert (v['selected_direction'],v['selected_factor'])==((first['direction'],first['factor']) if first else ('identity',0.))
   if first:assert v['applied_static']==first['static_loss'] and v['applied_full']==first['full_loss']
  else:assert (v['selected_direction'],v['selected_factor'])==('full_saved_momentum',1.)
  assert v['identity_static']==v['baseline_static'] and abs(v['identity_full']-v['baseline_full'])<=1e-6*max(1,abs(v['baseline_full']))
 parent=torch.load(plan['parent'],weights_only=False,map_location='cpu') if 'parent' in plan else None
 rows.append(dict(arm=arm,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),updates=updates,points=14816*updates,directions=[(v['selected_direction'],v['selected_factor']) for v in h]))
# Parent provenance comes from the original source artifact used by this workflow.
parent_path=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/standing_wm_calibration_v1/fit/world_model_calibrated.pt')
parent=torch.load(parent_path,weights_only=False,map_location='cpu')['world_model'];frozen=[]
for key,value in parent.items():
 if not key.startswith('net.'):
  for x in artifacts.values():assert torch.equal(x['world_model'][key],value),key
  frozen.append(key)
assert len(frozen)==22
reference=torch.load(a.reference_control,weights_only=False,map_location='cpu')
assert all(torch.equal(v,reference['world_model'][k]) for k,v in artifacts['control']['world_model'].items())
for key in ['native_replay_history.json','native_h32_history.json']:
 hs=[json.loads((a.root/arm/a.stage/key).read_text()) for arm in ['control','candidate']]
 assert [[z['indices'] for z in h] for h in hs][0]==[[z['indices'] for z in h] for h in hs][1]
assert [v['static_indices'] for v in histories['control']]==[v['static_indices'] for v in histories['candidate']]
out=a.root/(a.stage+'_integrity.json');assert not out.exists();out.write_text(json.dumps(dict(passed=True,models=rows,frozen_keys=frozen,control_bitexact=True,same_indices=True,policy_acceptance=False,scope='Implementation and same-fit finite probes only; independent validation still required'),indent=2));print('DUAL_LOSS_INTEGRITY_PASS',updates)
