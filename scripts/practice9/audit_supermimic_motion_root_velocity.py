"""Verify original motion source isolation and paired root velocity branches."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();r=a.collection;plan=json.loads((r/'plan.json').read_text());result=json.loads((r/'results.json').read_text());assert plan['smoke']==result['smoke']
source_root=Path(plan['source_collection']) if 'source_collection' in plan else Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_collection_cache_v2/full')
source=json.loads((source_root/'results.json').read_text())['rows'];assert source==plan['source_files']
source_manifest=Path(plan['source_manifest']) if 'source_manifest' in plan else source_root.parent/'plan.json'
assert plan['source_plan_sha256']==hashlib.sha256(source_manifest.read_bytes()).hexdigest()
info=json.loads((source_root/'input.json').read_text());assert plan['asset_sha256']==info['asset_sha256'] and plan['physics']==info['physics']
if 'source_manifest' in plan:assert info['plan_sha256']==plan['source_plan_sha256']
assert json.loads((r/'control_mapping.json').read_text())==json.loads((source_root/'control_mapping.json').read_text())
assert plan['num_envs']==(1 if plan.get('serial',False) else 12)
assert plan['levels']==[1.,2.] and plan['base_delta']==[.005,.005,.005,.02,.02,.02]
families={s:{x['source_family'] for x in source if x['split']==s} for s in ['fit','validation']};assert len(families['fit'])==len(families['validation'])==4 and not families['fit']&families['validation']
selected=[x for x in source if x['split']=='fit'][:1] if plan['smoke'] else source
expected=[];loaded={};files=[];counts=dict(write_failures=0,repeat_state_failures=0,repeat_next_failures=0,terminal_repeat_mismatches=0,nominal_replay_failures=0,nominal_terminal_mismatches=0);failures=[]
for s in selected:
 path=source_root/s['trace'];assert hashlib.sha256(path.read_bytes()).hexdigest()==s['sha256']
 with np.load(path,allow_pickle=False) as z:original={k:z[k].copy() for k in ['state','action','context','next_state','terminated','truncated']}
 indices=[0] if plan['smoke'] else sorted({0,s['steps']//2,s['steps']-1})
 for i in indices:
  for level,assignment in [(0.,0),(1.,0),(1.,1),(2.,0),(2.,1)]:
   expected.append((s['trace'],i,level,assignment))
   row=result['rows'][len(expected)-1];assert (row['source_trace'],row['index'],row['level'],row['assignment'])==expected[-1]
   assert row['split']==s['split'] and row['source_family']==s['source_family'] and row['count']==12
   path=(r/row['file']).resolve();assert path.is_relative_to(r.resolve());sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==row['sha256']
   with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
   if plan.get('restore_native_state'):
    with np.load(source_root/s['trace'],allow_pickle=False) as z:
     for key,width in [('native_qpos',37),('native_qvel',36),('native_qacc_warmstart',36)]:
      assert d[key].shape==(12,width)
      if key!='native_qvel' or level==0:np.testing.assert_array_equal(d[key],np.repeat(z[key][i:i+1],12,0))
      else:np.testing.assert_array_equal(d[key][:,6:],np.repeat(z[key][i:i+1,6:],12,0))
   assert all(np.isfinite(v).all() for v in d.values())
   for k,width in [('state',73),('desired',73),('action',30),('context',34),('next_state',73)]:assert d[k].shape==(12,width)
   signs=np.tile([-1.,1.],6)*(1 if assignment==0 else -1)
   delta=np.zeros((12,6),dtype=np.float32);delta[np.arange(12),np.repeat(np.arange(6),2)]=signs*np.repeat(plan['base_delta'],2)*level
   np.testing.assert_allclose(d['delta'],delta,atol=1e-8,rtol=0)
   desired=np.repeat(original['state'][i:i+1],12,0);desired[:,7:13]+=delta
   np.testing.assert_array_equal(d['desired'],desired)
   np.testing.assert_array_equal(d['action'],np.repeat(original['action'][i:i+1],12,0));np.testing.assert_allclose(d['context'],np.repeat(original['context'][i:i+1],12,0),atol=1e-7,rtol=0)
   write=float(np.abs(d['state']-desired).max());assert write==row['write_error'];counts['write_failures']+=write>2e-6
   if level==0:
    nominal_error=float(np.abs(d['next_state']-original['next_state'][i]).max());counts['nominal_replay_failures']+=nominal_error>1e-5
    counts['nominal_terminal_mismatches']+=int(np.count_nonzero(d['terminated']!=original['terminated'][i])+np.count_nonzero(d['truncated']!=original['truncated'][i]))
    if nominal_error>1e-5:failures.append(dict(kind='nominal_replay',file=row['file'],max_error=nominal_error))
   loaded[(s['trace'],i,level,assignment)]=d;files.append(dict(file=row['file'],sha256=sha,split=s['split']))
  for level in [1.,2.]:
   a0,a1=[loaded[(s['trace'],i,level,j)] for j in [0,1]];swap=np.arange(12).reshape(6,2)[:,::-1].reshape(-1)
   counts['repeat_state_failures']+=float(np.abs(a0['state']-a1['state'][swap]).max())>2e-6
   repeat_error=float(np.abs(a0['next_state']-a1['next_state'][swap]).max());counts['repeat_next_failures']+=repeat_error>1e-5
   if repeat_error>1e-5:failures.append(dict(kind='same_sign_repeat',trace=s['trace'],index=i,level=level,max_error=repeat_error))
   counts['terminal_repeat_mismatches']+=int(np.count_nonzero(a0['terminated']!=a1['terminated'][swap])+np.count_nonzero(a0['truncated']!=a1['truncated'][swap]))
assert len(expected)==len(result['rows'])==(5 if plan['smoke'] else 480)
assert result['endpoints']==len(expected)*12 and result['write_failures']==counts['write_failures']
out=dict(passed=not any(counts.values()),smoke=plan['smoke'],batches=len(expected),endpoints=result['endpoints'],counts=counts,failures=failures,files=files,source_families={k:sorted(v) for k,v in families.items()},scope='Branch archive/mapping/source/repeat checks only; all terminal outputs retained. Acquisition may read validation data; no optimizer or policy acceptance.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['files','source_families']}))
