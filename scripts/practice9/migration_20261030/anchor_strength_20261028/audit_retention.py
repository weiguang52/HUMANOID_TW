from pathlib import Path
import sys,json,numpy as np
state=Path(sys.argv[1]);d=state/'retention';jobs=json.loads((d/'jobs.json').read_text());log=(d/'run.log').read_text();rows=[]
assert (d/'jobs.json.done').read_text().strip()=='completed' and len(jobs)==2
for j in jobs:
 p=Path(j['output']);assert 'BATCH_COMPLETED '+str(p) in log
 x=np.load(p.with_suffix('.npz'));gold=np.load(state.parent/'anchor_strength_20261028/accepted_baseline/retention'/p.with_suffix('.npz').name)
 blend=np.load(p.with_suffix('.blend.npz'));v=blend['speed_and_candidate_weight'];meta=json.loads(p.read_text())
 exact={k:bool(np.array_equal(x[k],gold[k])) for k in ['q','action','anchor_actual','terminal','reference_q']}
 full=len(x['terminal'])==1000 and not x['terminal'].any()
 drift=float(np.linalg.norm(x['anchor_actual'][-1,:2]-x['anchor_actual'][50,:2])*1000)
 old=float(np.linalg.norm(gold['anchor_actual'][-1,:2]-gold['anchor_actual'][50,:2])*1000)
 zero=bool(np.all(v[...,1]==0))
 rows.append({'id':p.stem,'full20s':bool(full),'drift_mm':drift,'accepted_drift_mm':old,'zero_candidate_weight':zero,'accepted_trajectory_exact':exact,'passed':bool(full and zero and all(exact.values()) and drift<=old+1e-6)})
out={'complete':True,'passed':all(v['passed'] for v in rows),'rows':rows,'scope':'Fixed full20s retention against accepted policy; no walking acceptance'}
(state/'retention_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if not out['passed']:raise SystemExit(2)
