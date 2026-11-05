"""Verify dynamic paired action labels against archived state, command and native replay."""
import argparse,hashlib,json,datetime
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
r=a.collection;plan=json.loads((r/'plan.json').read_text());assert plan['contract']=='motion_action_branches_v1' and plan['restore_native_state'] and plan['serial']
source=Path(plan['source_collection']);assert hashlib.sha256((source/'control_mapping.json').read_bytes()).hexdigest()==hashlib.sha256((r/'control_mapping.json').read_bytes()).hexdigest()
result=json.loads((r/'results.json').read_text());dense=json.loads((r/'dense_jacobian_diagnostic.json').read_text());assert dense['runtime_dense_verified'];assert result['write_failures']==0
source_rows={x['trace']:x for x in plan['source_files']};groups={};maxima={'zero_replay':0.,'state_write':0.,'context':0.,'action':0.};files=[]
for row in result['rows']:
 path=r/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];original=source_rows[row['source_trace']];sp=source/original['trace'];assert hashlib.sha256(sp.read_bytes()).hexdigest()==original['sha256'];assert row['split']==original['split']
 with np.load(path) as z:d={k:z[k] for k in z.files}
 with np.load(sp) as z:b={k:z[k] for k in z.files}
 index=row['index'];assert row['count']==60
 for k,width in [('state',73),('action',30),('context',34),('next_state',73)]:assert d[k].shape==(60,width) and np.isfinite(d[k]).all()
 delta=np.zeros((60,30),np.float32);delta[np.arange(60),np.repeat(np.arange(30),2)]=np.tile(np.array([-1,1],np.float32),30)*(1 if row['assignment']==0 else -1)*row['level']
 np.testing.assert_array_equal(d['delta'],delta);expected=b['action'][index]+delta;np.testing.assert_array_equal(d['action'],expected)
 maxima['state_write']=max(maxima['state_write'],float(abs(d['state']-b['state'][index]).max()));maxima['context']=max(maxima['context'],float(abs(d['context']-b['context'][index]).max()))
 assert maxima['state_write']<=2e-6 and maxima['context']<=1e-7
 for k in ('native_qpos','native_qvel','native_qacc_warmstart'):np.testing.assert_array_equal(d[k],np.broadcast_to(b[k][index],d[k].shape))
 if row['level']==0:
  maxima['zero_replay']=max(maxima['zero_replay'],float(abs(d['next_state']-b['next_state'][index]).max()));assert maxima['zero_replay']<=1e-5
 groups.setdefault((row['source_trace'],index),{})[(row['level'],row['assignment'])]=d
 files.append({'file':row['file'],'sha256':row['sha256'],'split':row['split']})
for key,g in groups.items():assert set(g)=={(0.,0),(.02,0),(.02,1),(.05,0),(.05,1)}
expected_groups=set()
selected=[x for x in plan['source_files'] if x['split']=='fit'][:1] if plan['smoke'] else plan['source_files']
for row in selected:
 with np.load(source/row['trace']) as z:n=len(z['state'])
 expected_groups.update((row['trace'],i) for i in ([0] if plan['smoke'] else sorted({0,n//2,n-1})))
assert set(groups)==expected_groups and len(result['rows'])==len(groups)*5
out={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'smoke':plan['smoke'],'groups':len(groups),'endpoints':len(files)*60,'max_errors':maxima,'files':files,'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),'policy_acceptance':False,'scope':'Paired one-step native label integrity only; not model quality, skill or hardware acceptance.'};a.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='files'}))
