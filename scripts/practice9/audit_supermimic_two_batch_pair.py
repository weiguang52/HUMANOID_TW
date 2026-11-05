"""Independent matched-fit implementation audit, not policy acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--stage',choices=['smoke4','fit'],required=True)
p.add_argument('--reference-control',type=Path,required=True)
p.add_argument('--updates',type=int,choices=[4,1000],required=True)
a=p.parse_args();updates=a.updates
artifacts={};histories={};models=[]
for arm in ['control','candidate']:
    root=a.root/arm/a.stage;path=root/'world_model_calibrated.pt'
    artifact=torch.load(path,map_location='cpu',weights_only=False);artifacts[arm]=artifact
    plan=artifact['calibration_plan'];enabled=arm=='candidate'
    assert artifact['calibration_updates']==updates and plan['two_direction_step']==enabled
    assert plan['actual_prediction_points_min']==plan['actual_prediction_points_max']==21056
    assert plan['actual_prediction_points_total']==21056*updates
    assert len(artifact['optimizer']['state'])==8
    assert all(int(v['step'])==updates for v in artifact['optimizer']['state'].values())
    history=json.loads((root/'two_direction_history.json').read_text());histories[arm]=history
    assert len(history)==updates
    for i,row in enumerate(history,1):
        assert row['update']==i and row['enabled']==enabled
        assert len(row['baseline'])==len(row['identity'])==18
        assert all(abs(v-b)<=1e-6*max(1,abs(b)) for v,b in zip(row['identity'],row['baseline']))
        assert len(row['weights'])==18 and all(math.isfinite(w) and w>=0 for w in row['weights'])
        assert abs(sum(row['weights'])-1)<1e-12
        assert [(v['direction'],v['factor']) for v in row['probes']]==[(n,f) for n in ['common'] for f in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]]
        for probe in row['probes']:
            assert len(probe['values'])==18
            assert probe['accepted']==all(v is not None and math.isfinite(v) and v<=b for v,b in zip(probe['values'],row['baseline']))
        choices=[next((v for v in row['probes'] if v['direction']==name and v['accepted']),None) for name in ['common']]
        choices=[v for v in choices if v is not None]
        first=min(choices,key=lambda v:v['values'][1]) if choices else None
        if enabled:
            assert row['selected_direction']==(first['direction'] if first else 'identity')
            assert row['selected_factor']==(first['factor'] if first else 0.)
            assert list(row['applied_values'])==(first['values'] if first else row['identity'])
        else:assert row['selected_factor']==1. and row['applied_values'] is None and row['selected_direction']=='adam'
    models.append({'arm':arm,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'updates':updates,'points':21056*updates,'factors':[v['selected_factor'] for v in history]})
parent_path=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/standing_wm_calibration_v1/fit/world_model_calibrated.pt')
parent=torch.load(parent_path,map_location='cpu',weights_only=False)['world_model'];frozen=[]
for key,value in parent.items():
    if not key.startswith('net.'):
        for artifact in artifacts.values():assert torch.equal(artifact['world_model'][key],value),key
        frozen.append(key)
assert len(frozen)==22
reference=torch.load(a.reference_control,map_location='cpu',weights_only=False)
assert artifacts['control']['world_model'].keys()==reference['world_model'].keys()
assert all(torch.equal(value,reference['world_model'][key]) for key,value in artifacts['control']['world_model'].items())
def equal(x,y):
    if isinstance(x,torch.Tensor):return torch.equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(v,w) for v,w in zip(x,y))
    return x==y
assert equal(artifacts['control']['optimizer'],reference['optimizer'])
for filename in ['native_replay_history.json','native_h32_history.json']:
    hs=[json.loads((a.root/arm/a.stage/filename).read_text()) for arm in ['control','candidate']]
    assert [[z['indices'] for z in h] for h in hs][0]==[[z['indices'] for z in h] for h in hs][1]
assert [v['static_indices'] for v in histories['control']]==[v['static_indices'] for v in histories['candidate']]
out=a.root/(a.stage+'_integrity.json');assert not out.exists()
assert all([v[field] for v in histories['control']]==[v[field] for v in histories['candidate']] for field in ['first_rng_sha256','second_rng_sha256'])
assert all(v['first_rng_sha256']!=v['second_rng_sha256'] for h in histories.values() for v in h)
out.write_text(json.dumps({'passed':True,'models':models,'frozen_keys':frozen,'control_weights_optimizer_bitexact':True,'same_indices':True,'policy_acceptance':False,'scope':'Implementation and finite two-fit-batch18 probes only; all original independent gates still required'},indent=2))
print('TWO_DIRECTION_INTEGRITY_PASS',updates)

