from pathlib import Path
import torch,json,hashlib,datetime
root=Path(__file__).parent;torch.set_num_threads(1)
for state in (root,root.parent/'anchor_strength_seed123_20261028'):
 cp=Path((state/'checkpoint').read_text().strip());freeze=json.loads((state/'checkpoint_freeze.json').read_text());assert hashlib.sha256(cp.read_bytes()).hexdigest()==freeze['sha256']
 actor=torch.load(cp,map_location='cpu',weights_only=False)['actor_state_dict'];path=cp.parent/'exported/policy.pt';jit=torch.jit.load(str(path),map_location='cpu').state_dict();assert set(actor)-set(jit)=={'distribution.std_param'} and set(jit)<=set(actor)
 checks={k:torch.equal(actor[k],v) for k,v in jit.items()};assert checks and all(checks.values());assert actor['mlp.0.weight'].shape[1]==203
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'checkpoint':str(cp),'checkpoint_sha256':freeze['sha256'],'jit':str(path),'jit_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parameter_checks':checks,'actor_input_dim':203,'excluded':'distribution.std_param: deterministic export omits sampling distribution','scope':'Exact deterministic MLP parameter binding; does not establish walking quality or visual acceptance.'};(state/'export_binding_audit.json').write_text(json.dumps(out,indent=2));print(state.name,'all',len(checks),'MLP tensors exact')
