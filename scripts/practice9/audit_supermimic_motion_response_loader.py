"""Audit actual fit-only branch array reads and matched condition structure."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic_motion_root_response import load_motion_root_response,derivative
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--collection',type=Path,action='append',required=True)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();cfg=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['cfg'];reports=[]
for root in a.collection:
 reads=[];original=np.load
 def tracked(path,*args,**kwargs):
  reads.append(Path(path).resolve());return original(path,*args,**kwargs)
 np.load=tracked
 try:data,den,axes,meta=load_motion_root_response(root,cfg,split='fit',device='cpu')
 finally:np.load=original
 assert len(reads)==len(set(reads))==192
 expected={(root/x['file']).resolve() for x in meta['files']};assert set(reads)==expected
 rows=json.loads((root/'results.json').read_text())['rows'];selected=[x for x in rows if (root/x['file']).resolve() in expected]
 assert len(selected)==192 and all(x['split']=='fit' and x['level']>0 for x in selected)
 assert len({(x['source_trace'],x['index']) for x in selected})==48
 assert len({x['source_family'] for x in selected})==4
 assert meta['conditions']==576 and not meta['validation_arrays_loaded']
 for key,width in [('state',73),('action',30),('context',34),('next_state',73)]:assert tuple(data[key].shape)==(576,4,width) and torch.isfinite(data[key]).all()
 assert tuple(den.shape)==(576,1) and tuple(axes.shape)==(576,)
 assert all(int((axes==k).sum())==96 for k in range(6))
 actual=derivative(data['state'],den);expected_axes=torch.eye(6)[axes]
 # Derived from four input-write errors each bounded by the existing2e-6 gate.
 bound=8e-6/float(den.min())
 error=float((actual-expected_axes).abs().max());assert error<=bound
 reports.append(dict(collection=str(root.resolve()),files_loaded=192,fit_families=4,anchors=48,conditions=576,source_meta=meta,input_derivative_identity_max_error=error,derived_input_write_error_bound=bound,validation_arrays_loaded=False))
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha
report=dict(passed=True,checkpoint_sha256=sha,collections=reports,optimizer_updates=0,scope='CPU fit-loader/provenance/order integrity only; no model-quality or policy acceptance')
a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='collections'}))
