"""Independently verify fixed query coverage, archives and native error metrics."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();r=a.root;plan=json.loads((r/'plan.json').read_text());result=json.loads((r/'results.json').read_text());assert not plan['smoke'] and not result['smoke']
collection=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_collection_cache_v2/full')
sources=[x for x in json.loads((collection/'results.json').read_text())['rows'] if x['split']=='fit'];assert len(sources)==16
expected=[];data={}
for row in sources:
 path=collection/row['trace'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 with np.load(path,allow_pickle=False) as z:data[row['trace']]={k:z[k].copy() for k in ['state','action','context','next_state']}
for arm in ['control','candidate']:
 for source in sources:
  for index in sorted({0,7,min(31,source['steps']-1)}):
   for mode in ['true','free','root_velocity_true']:expected.append((arm,source['trace'],index,mode))
assert len(expected)==plan['query_count']==result['queries']==288
assert [(x['arm'],x['trace'],x['index'],x['mode']) for x in result['rows']]==expected
records={};errors={};native_outputs={};counts=dict(write_failures=0,repeat_failures=0,true_replay_failures=0)
for row in result['rows']:
 path=(r/row['file']).resolve();assert path.is_relative_to(r.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 assert all(np.isfinite(v).all() for v in d.values())
 for k,shape in [('desired',(1,73)),('restored',(2,73)),('context',(2,34)),('action',(1,30)),('native',(2,73)),('model',(2,73)),('archived',(1,73))]:assert d[k].shape==shape
 source=data[row['trace']];i=row['index']
 np.testing.assert_array_equal(d['action'][0],source['action'][i]);np.testing.assert_array_equal(d['archived'][0],source['next_state'][i]);np.testing.assert_allclose(d['context'],np.repeat(source['context'][i:i+1],2,0),atol=1e-7,rtol=0)
 write=float(np.abs(d['restored']-d['desired']).max());repeat=float(np.abs(d['native'][0]-d['native'][1]).max())
 assert write==row['write_error'] and repeat==row['repeat_error']
 counts['write_failures']+=write>2e-6;counts['repeat_failures']+=repeat>1e-5
 if row['mode']=='true':
  np.testing.assert_array_equal(d['desired'][0],source['state'][i]);e=float(np.abs(d['native'][0]-d['archived'][0]).max());assert e==row['archived_error'];counts['true_replay_failures']+=e>1e-5
 records[(row['arm'],row['trace'],i,row['mode'])]=d
 key=(row['arm'],row['mode'],i);errors.setdefault(key,[]).append(d['model'][0]-d['native'][0])
 native_outputs.setdefault(key,[]).append(d['native'][0])
for arm,trace,i in {(k[0],k[1],k[2]) for k in records}:
 true,free,mixed=[records[(arm,trace,i,mode)] for mode in ['true','free','root_velocity_true']]
 expected_mixed=free['desired'].copy();expected_mixed[:,7:13]=true['desired'][:,7:13];np.testing.assert_array_equal(expected_mixed,mixed['desired'])
 if i==0:np.testing.assert_array_equal(true['desired'],free['desired'])
for k,v in counts.items():assert v==result[k]
sl=[('linear',slice(7,10)),('angular',slice(10,13)),('joint_velocity',slice(43,73))]
metrics=[]
for (arm,mode,index),pieces in errors.items():
 e=np.stack(pieces).astype(np.float64);metrics.append(dict(arm=arm,mode=mode,index=index,queries=len(e),rmse={k:float(np.sqrt(np.mean(e[:,s]**2))) for k,s in sl}))
decomposition=[]
for arm in ['control','candidate']:
 for mode in ['true','free','root_velocity_true']:
  for i in sorted({key[2] for key in records}):
   pool=[d for (aa,trace,j,mm),d in records.items() if aa==arm and mm==mode and j==i]
   model_error=np.stack([d['model'][0]-d['native'][0] for d in pool]).astype(np.float64)
   physical_shift=np.stack([d['native'][0]-d['archived'][0] for d in pool]).astype(np.float64)
   total=model_error+physical_shift
   decomposition.append(dict(arm=arm,mode=mode,index=i,queries=len(pool),metrics={k:dict(model_native_rmse=float(np.sqrt(np.mean(model_error[:,s]**2))),native_original_next_rmse=float(np.sqrt(np.mean(physical_shift[:,s]**2))),model_original_next_rmse=float(np.sqrt(np.mean(total[:,s]**2))),mse_cross_term=float(2*np.mean(model_error[:,s]*physical_shift[:,s]))) for k,s in sl}))
responses=[]
for arm in ['control','candidate']:
 for i in sorted({k[2] for k in records}):
  pairs=[(records[(arm,t,j,'free')],records[(arm,t,j,'root_velocity_true')]) for aa,t,j,mode in records if aa==arm and j==i and mode=='free']
  e=np.stack([(m['model'][0]-f['model'][0])-(m['native'][0]-f['native'][0]) for f,m in pairs]).astype(np.float64)
  responses.append(dict(arm=arm,index=i,queries=len(pairs),finite_response_error_rmse={k:float(np.sqrt(np.mean(e[:,s]**2))) for k,s in sl}))
out=dict(archive_and_coverage_passed=True,queries=288,counts=counts,model_native_metrics=metrics,root_velocity_transplant_response=responses,error_decomposition=decomposition,weights_unchanged_reported=result['weights_unchanged'],scope='Fixed fit-only query archive verification; precision failures preserved. Finite response across different restored hybrid states, not infinitesimal Jacobian, policy stability or universal physics proof.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
