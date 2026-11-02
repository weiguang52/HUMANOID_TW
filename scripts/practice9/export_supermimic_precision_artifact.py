"""Export a new explicitly declared arithmetic artifact; never mutate source or fit."""
import argparse,copy,hashlib,json
from pathlib import Path
import torch
from supermimic_precision_contract import metadata,state_digest,validate
from supermimic_yaw_canonical import KIND,coordinate_metadata
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();source_sha=hashlib.sha256(a.source.read_bytes()).hexdigest();c=torch.load(a.source,map_location='cpu',weights_only=False)
assert c['kind']==KIND and c['coordinate_contract']['enabled'] and 'precision_contract' not in c
original=copy.deepcopy(c['world_model']);contract=metadata(source_sha,state_digest(original))
c['precision_contract']=contract;c['offline_only']=True;c['resumable_training']=False;c['policy_promotion_approved']=False;c.pop('optimizer',None)
c['calibration_plan']=dict(c['calibration_plan'],precision_contract=contract)
c['coordinate_contract']=coordinate_metadata(True,network_dtype='float64');c['calibration_plan']['coordinate_contract']=dict(c['coordinate_contract'])
assert validate(c) and all(torch.equal(v,c['world_model'][k]) for k,v in original.items())
a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(c,a.output)
assert hashlib.sha256(a.source.read_bytes()).hexdigest()==source_sha
print(json.dumps(dict(output=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),source_sha256=source_sha,stored_weights_unchanged=True,optimizer_updates=0,precision_contract=contract,promotion_approved=False)),flush=True)
