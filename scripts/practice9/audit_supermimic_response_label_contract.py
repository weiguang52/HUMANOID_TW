"""Fit-only action-response label, cancellation and amplitude consistency audit."""
import argparse,hashlib,json,datetime,ast
from pathlib import Path
import numpy as np
import torch
from supermimic_antithetic_response import AntitheticResponseLoss
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--integrity',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
l=AntitheticResponseLoss(a.root,a.integrity,device='cpu');artifact=torch.load(a.parent,map_location='cpu',weights_only=False);s=artifact['world_model'];print('normalization_keys',[k for k in s if k.startswith('state_norm.')],flush=True)
var=s['state_norm.var'].double();mask=s['state_norm.frozen_mask'];norm_source=Path('third_party/supermimic_mjlab/src/supermimic/algo/networks.py');tree=ast.parse(norm_source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RunningNorm');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');eps=ast.literal_eval(init.args.kw_defaults[[x.arg for x in init.args.kwonlyargs].index('eps')]);std=(var+eps).sqrt();std=torch.where(mask,torch.ones_like(std),std)
plan=json.loads((a.root/'plan.json').read_text());assert plan['epsilons']==[.02,.05]
expected=torch.eye(30).repeat(len(plan['fit_warmups'])*len(plan['epsilons']),1)
assert torch.equal(l.derivative(l.data['action']),expected)
assert torch.equal(l.derivative(l.data['context']),torch.zeros(240,34))
vd=l.data['next_state'].double();target64=(vd[:,1]-vd[:,0]+vd[:,2]-vd[:,3])/l.den.double();target32=l.target.double()
residual=l.derivative(l.data['state']).double()/std
out={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fit_only':True,'conditions':240,'normalizer_epsilon':eps,'action_derivative_exact_onehot':True,'context_cancellation_exact':True,'within_assignment_state_absmax':float((l.data['state'][:,0]-l.data['state'][:,1]).abs().max()),'normalized_initial_state_derivative_rms':float(residual.square().mean().sqrt()),'groups':{},'policy_acceptance':False,'limitation':'State residual is a diagnostic, not a proven bound on output error. Amplitude disagreement can reflect real contact nonlinearity; labels are not replaced.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.integrity,a.parent,a.root/'plan.json',norm_source]}}
for name,sl in [('root_angular',slice(10,13)),('joint_velocity',slice(43,73))]:
 t=target64[:,sl]/std[sl];roundoff=(target32[:,sl]-target64[:,sl])/std[sl];pair=t.reshape(4,2,30,-1);diff=pair[:,0]-pair[:,1]
 out['groups'][name]={'target_normalized_rms':float(t.square().mean().sqrt()),'float32_subtraction_error_rms':float(roundoff.square().mean().sqrt()),'epsilon_difference_rms':float(diff.square().mean().sqrt()),'epsilon_cosine':float(torch.nn.functional.cosine_similarity(pair[:,0].flatten(),pair[:,1].flatten(),dim=0))}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out))
