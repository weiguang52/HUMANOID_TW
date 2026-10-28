"""Repeated seed42 initial/matrix replay parity; no policy quality acceptance."""
from pathlib import Path
import argparse,json,numpy as np
parser=argparse.ArgumentParser();parser.add_argument('state',type=Path);parser.add_argument('--require_complete',action='store_true');args=parser.parse_args()
s=args.state;matrix=s.parent/(s.name.replace('_20261026','_eval_20261026'))/'candidate'
initial=s/'evaluation/actual';rows=[];pending=[]
log=(matrix/'seed42.log').read_text() if (matrix/'seed42.log').exists() else ''
for j in json.loads((matrix/'jobs42.json').read_text()):
 p=Path(j['output']);q=initial/p.name
 paths=[p,q,p.with_suffix('.npz'),q.with_suffix('.npz'),p.with_suffix('.blend.npz'),q.with_suffix('.blend.npz')]
 if 'BATCH_COMPLETED '+str(p) not in log or not all(x.exists() for x in paths):pending.append(p.name);continue
 a=np.load(p.with_suffix('.npz'));b=np.load(q.with_suffix('.npz'))
 checks={k:bool(np.array_equal(a[k],b[k])) for k in ['q','action','anchor_actual','reference_q','terminal']}
 c=np.load(p.with_suffix('.blend.npz'));d=np.load(q.with_suffix('.blend.npz'))
 checks['blend_trace']=c.files==d.files and all(np.array_equal(c[k],d[k]) for k in c.files)
 checks['saved_training_contract']=json.loads(p.read_text()).get('training_action_filter_contract')==json.loads(q.read_text()).get('training_action_filter_contract')
 rows.append({'id':p.stem,'checks':checks,'passed':all(checks.values())})
out={'observed':len(rows),'expected':14,'complete':len(rows)==14 and not pending,'all_observed_exact':bool(rows) and all(v['passed'] for v in rows),'rows':rows,'pending':pending,'scope':'Repeated evaluation seed42 initial vs matrix only, not other seeds or quality acceptance'}
(s/'initial_matrix_parity.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k not in ['rows','pending']})
assert not rows or out['all_observed_exact']
if args.require_complete:assert out['complete'] and out['all_observed_exact']
