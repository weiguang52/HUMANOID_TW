"""Verify every final heading CV artifact and report all held-out folds."""
import argparse,json,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'independent_artifact_audit.json';assert not out.exists();result=json.loads((r/'result.json').read_text());assert len(result['folds'])==4
records=[];folds=[];seen=set()
for fold in result['folds']:
 held=set(fold.get('held_out_families',[fold['held_out_family']]));train=set(fold['train_families']);assert not held&train and not seen&held;seen|=held
 for arm in ['constant','heading']:
  path=Path(fold['arms'][arm]['artifact']);c=torch.load(path,map_location='cpu',weights_only=False)
  assert c['updates']==1000 and c['lr']==5e-4 and c['heading_enabled']==(arm=='heading') and c['response_only']==result['response_only']
  assert set(c.get('held_out_families',[c['held_out_family']]))==held and set(c['training_families'])==train and c['batch_sha256']==fold['batch_sha256']
  assert len(c['optimizer']['state'])==6 and all(int(v['step'])==1000 for v in c['optimizer']['state'].values())
  assert all(torch.isfinite(v).all() for v in c['state_dict'].values()) and not c['promotion_approved'] and not c['resumable_policy_training']
  assert hashlib.file_digest(Path(c['source_checkpoint']).open('rb'),'sha256').hexdigest()==c['source_sha256']==result['source_sha256']
  records.append(dict(file=str(path),sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()))
 constant=fold['arms']['constant']['held_out']['scaled_mse'];heading=fold['arms']['heading']['held_out']['scaled_mse'];folds.append(dict(fold=fold['held_out_family'],constant=constant,heading=heading,heading_over_constant=heading/constant if constant else None,heading_improves=heading<constant))
passed=all(f['heading_improves'] for f in folds);assert passed==result['all_four_heading_improve']
report=dict(artifact_integrity_passed=True,all_final_artifacts=records,source_family_isolation=True,folds=folds,all_four_heading_improve=passed,decision='requires_absolute_horizon_validation' if passed else 'rejected_source_generalization_regression',policy_acceptance=False)
out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='all_final_artifacts'}))
