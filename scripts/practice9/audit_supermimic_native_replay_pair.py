"""Check native-replay paired budgets, frozen parameters and zero-control replay."""
import argparse,json,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
for key in ['root','parent','reference-control','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--h32',action='store_true');p.add_argument('--updates',type=int,choices=[4,1000],required=True);a=p.parse_args();assert not a.output.exists();stage='smoke' if a.updates==4 else 'fit';parent=torch.load(a.parent,map_location='cpu',weights_only=False);reference=torch.load(a.reference_control,map_location='cpu',weights_only=False);histories=[];reports=[]
for arm,coefficient in [('control',False),('candidate',True)]:
 f=a.root/arm/stage/'world_model_calibrated.pt';c=torch.load(f,map_location='cpu',weights_only=False);plan=c['calibration_plan'];w=c['world_model'];assert c['calibration_updates']==a.updates and plan['native_replay_loss']==(True if a.h32 else coefficient)
 assert plan['actual_prediction_points_min']==plan['actual_prediction_points_max']==plan['prediction_endpoints_per_update']==784 and plan['actual_prediction_points_total']==784*a.updates
 assert plan['native_replay_generator']==55 and plan['native_replay_input']['states']==4515 and len(plan['native_replay_input']['fit_families'])==24 and not plan['native_replay_input']['validation_arrays_loaded']
 if a.h32:
  assert plan['native_h32_mix']==coefficient and plan['native_h32_generator']==56 and plan['native_h32_input']['windows']==1588
  assert plan['native_h32_short_batch']==(60 if coefficient else 64) and plan['native_h32_long_batch']==int(coefficient)
  long_history=json.loads((a.root/arm/stage/'native_h32_history.json').read_text());assert len(long_history)==a.updates
  indices=[x['indices'] for x in long_history];assert all(len(x)==1 and 0<=x[0]<1588 for x in indices)
  if arm=='control':control_long_indices=indices
  else:assert indices==control_long_indices
 frozen=[k for k in w if not k.startswith('net.')];assert frozen and all(torch.equal(w[k],parent['world_model'][k]) for k in frozen)
 assert all(torch.isfinite(v).all() for v in w.values()) and len(c['optimizer']['state'])==8 and all(int(x['step'])==a.updates for x in c['optimizer']['state'].values())
 if arm=='control':assert set(w)==set(reference['world_model']) and all(torch.equal(w[k],reference['world_model'][k]) for k in w), 'Zero-coefficient control did not replay exactly'
 history=json.loads((a.root/arm/stage/'native_replay_history.json').read_text());assert len(history)==a.updates and all(len(x['indices'])==16 and all(0<=i<4515 for i in x['indices']) for x in history);histories.append([x['indices'] for x in history])
 reports.append(dict(arm=arm,sha256=hashlib.file_digest(f.open('rb'),'sha256').hexdigest(),frozen_keys=len(frozen),optimizer_states=8,updates=a.updates))
assert histories[0]==histories[1]
a.output.write_text(json.dumps(dict(passed=True,native_h32=a.h32,updates=a.updates,models=reports,zero_control_bitexact=True,same_native_indices=True,points_per_update=784,policy_acceptance=False,scope='Implementation/budget/frozen-source audit only; no prediction or control acceptance.'),indent=2));print('NATIVE_PAIR_INTEGRITY_PASS',a.updates)
