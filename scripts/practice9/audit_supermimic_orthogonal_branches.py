"""Audit the entire fixed branch grid and swapped-world numerical sensitivity."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);a=p.parse_args();root=a.collection;assert not (root/'integrity.json').exists()
plan=json.loads((root/'plan.json').read_text());report=json.loads((root/'results.json').read_text());assert report['contract']==plan['contract']=='branch_transitions_not_episodes_v1'
expected={(split,w,kind,eps,assignment) for split,ws in [('fit',plan['fit_warmups']),('validation',plan['validation_warmups'])] for w in ws for kind,eps,assignment in [('zero',None,None)]+[('branch',e,b) for e in plan['epsilons'] for b in range(2)]}
rows={(r['split'],r['warmup'],r['kind'],r.get('epsilon'),r.get('assignment')):r for r in report['rows']};assert rows.keys()==expected and len(report['rows'])==40
mapping=json.loads((root/'control_mapping.json').read_text());assert len(mapping['state_joint_names'])==len(mapping['action_joint_names'])==30
loaded={};provenance=[]
for key,row in rows.items():
 path=(root/row['file']).resolve();assert path.is_relative_to(root.resolve())
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 for name,width in [('state',73),('next_state',73),('action',30),('context',34)]:assert d[name].shape==(60,width) and np.isfinite(d[name]).all()
 assert d['terminated'].shape==(60,) and not d['terminated'].any(), 'Preserve failed branches, do not fit silently'
 if row['kind']=='zero':assert not np.any(d['action'])
 else:
  assert d['joint'].tolist()==np.repeat(np.arange(30),2).tolist()
  assert d['sign'].tolist()==(np.tile([-1,1],30)*(1 if row['assignment']==0 else -1)).tolist()
  wanted=np.zeros((60,30),np.float32);wanted[np.arange(60),d['joint']]=d['sign']*row['epsilon'];np.testing.assert_array_equal(d['action'],wanted)
  assert row['max_replay_state_error']<=plan['replay_atol']
 loaded[key]=d;provenance.append(dict(file=row['file'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
def compare(x,y):
 x=x.astype(np.float64).reshape(-1);y=y.astype(np.float64).reshape(-1);den=np.linalg.norm(x)*np.linalg.norm(y)
 return dict(norm_assignment0=float(np.linalg.norm(x)),norm_assignment1=float(np.linalg.norm(y)),cosine=float(x@y/den) if den>1e-15 else None,rmse=float(np.sqrt(np.mean((x-y)**2))))
checks=[]
for split,ws in [('fit',plan['fit_warmups']),('validation',plan['validation_warmups'])]:
 for w in ws:
  zero=loaded[(split,w,'zero',None,None)]
  for eps in plan['epsilons']:
   deriv=[];pairdiff=[]
   for assignment in range(2):
    d=loaded[(split,w,'branch',eps,assignment)];np.testing.assert_array_equal(d['zero_state'],zero['state']);np.testing.assert_array_equal(d['zero_next_state'],zero['next_state']);np.testing.assert_allclose(d['context'],zero['context'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(d['state'],zero['state'],atol=plan['replay_atol'],rtol=0)
    delta=d['next_state'].astype(np.float64)-zero['next_state'].astype(np.float64)
    value=(delta[1::2]-delta[::2])/(2*eps)*(1 if assignment==0 else -1);deriv.append(value)
    pairdiff.append(float(np.max(np.abs(d['state'][1::2]-d['state'][::2]))))
   metrics={}
   for name,sl in [('root_position',slice(0,3)),('root_linear_velocity',slice(7,10)),('root_angular_velocity',slice(10,13)),('joint_position',slice(13,43)),('joint_velocity',slice(43,73))]:metrics[name]=compare(deriv[0][:,sl],deriv[1][:,sl])
   checks.append(dict(split=split,warmup=w,epsilon=eps,pair_initial_state_max_difference=pairdiff,swapped_assignment_sensitivity=metrics))
out=dict(passed=True,contract=plan['contract'],fit_branches=960,validation_branches=960,zero_responses=480,files=provenance,checks=checks,max_replay_error=max(r.get('max_replay_state_error',0) for r in report['rows']),scope='Branch transitions, never contiguous episodes. Static temporal split, not independent-motion validation. Swapped-world numerical comparison is reported, not a learned-skill acceptance gate.')
(root/'integrity.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in out.items() if k not in ['checks','files']}))
