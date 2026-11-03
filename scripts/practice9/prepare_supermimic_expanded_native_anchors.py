"""Freeze first/middle/last states of every expanded native trace."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
r=a.root;collection=r/'native_full';audit=json.loads((r/'full_independent_audit.json').read_text());assert audit['passed'] and not audit['smoke']
manifest=r/'collection_plan.json';source_plan=json.loads(manifest.read_text());assert source_plan['contract']=='expanded_native_contact_sources_v1'
results=json.loads((collection/'results.json').read_text())['rows'];assert len(results)==96
records={k:[] for k in ['query_state','source_state','source_next_state','action','context','source_native_qpos','source_native_qvel','source_native_warmstart']};rows=[]
for trace in results:
 f=collection/trace['trace'];assert hashlib.sha256(f.read_bytes()).hexdigest()==trace['sha256']
 with np.load(f,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 n=len(d['state']);assert n>=3
 for index in [0,(n-1)//2,n-1]:
  for key,value in [('query_state',d['state'][index]),('source_state',d['state'][index]),('source_next_state',d['next_state'][index]),('action',d['action'][index]),('context',d['context'][index]),('source_native_qpos',d['native_qpos'][index]),('source_native_qvel',d['native_qvel'][index]),('source_native_warmstart',d['native_qacc_warmstart'][index])]:records[key].append(value)
  rows.append(dict(query=len(rows),depth=0,trace=trace['trace'],id=trace['id'],source_family=trace['source_family'],start=trace['frame'],index=index,reference_frame=trace['frame']+index,split='fit',source_original_termination=bool(trace['termination_reasons'])))
assert len(rows)==288 and len({x['source_family'] for x in rows})==24
a.output.mkdir(parents=True);f=a.output/'queries.npz';np.savez_compressed(f,**{k:np.stack(v) for k,v in records.items()})
out=dict(contract='expanded_fit_native_anchor_v1',queries=288,rows=rows,sources=[dict(file=str(collection/x['trace']),sha256=x['sha256']) for x in results],collection=str(collection),manifest=str(manifest),query_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),collection_audit_sha256=hashlib.sha256((r/'full_independent_audit.json').read_bytes()).hexdigest(),selection='Every fixed trace first/middle/last input state, no quality/termination filtering.',validation_arrays_loaded=False,training_approved=False,promotion_approved=False)
(a.output/'query_plan.json').write_text(json.dumps(out,indent=2));print('FROZEN_NATIVE_ANCHORS',len(rows),out['query_sha256'])
