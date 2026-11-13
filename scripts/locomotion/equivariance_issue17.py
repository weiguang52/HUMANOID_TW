"""Deterministic CPU action equivariance on one frozen, unfiltered state set."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from issue17_symmetry import TrainingMirror
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4)
 manifest=json.loads((ROOT/'fixed_state_set.json').read_text());assert manifest['count']==102400 and manifest['frozen_before_new_model_comparison']
 states=[];sources=[]
 for row in manifest['files']:
  p=Path(row['path']);assert sha(p)==row['sha256'];z=np.load(p);assert z['actor'].shape==(20,1024,101)
  states.append(z['actor'].reshape(-1,101));sources.extend([row['training_seed']]*20480)
 states=np.concatenate(states);assert states.shape==(102400,101)
 contract_path=Path('validation_artifacts/stage1_issue16_20261111/mirror/moments/control6000_s1_eval1/mirror_runtime_contract.json')
 contract=json.loads(contract_path.read_text());mapping=TrainingMirror(contract['joint_names'],contract['action_names'],contract['action_scales'])
 cohort=json.loads((ROOT/'cohort.json').read_text());models=[dict(group='control6000',training_seed=i,**{k:cohort['models'][f'control6000_s{i}'][k] for k in ['checkpoint','sha256']}) for i in range(1,6)]
 trained=json.loads((ROOT/'training_records.json').read_text());assert len(trained)==10;models+=trained
 folder=ROOT/'equivariance';folder.mkdir(exist_ok=True);results=[]
 for record in models:
  checkpoint=Path(record['checkpoint']);assert sha(checkpoint)==record['sha256']
  dummy=TensorDict({'policy':torch.zeros(1,101)},batch_size=[1])
  actor=MLPModel(dummy,{'actor':['policy']},'actor',30,hidden_dims=[512,256,128],activation='elu',obs_normalization=False,distribution_cfg={'class_name':'GaussianDistribution','init_std':.5,'std_type':'scalar'})
  loaded=torch.load(checkpoint,map_location='cpu',weights_only=False);actor.load_state_dict(loaded['actor_state_dict'],strict=True);actor.eval()
  errors=[];norms=[]
  with torch.inference_mode():
   for start in range(0,len(states),1024):
    x=torch.from_numpy(states[start:start+1024]);o=TensorDict({'policy':x},batch_size=[len(x)]);m=TensorDict({'policy':mapping.actor(x)},batch_size=[len(x)])
    a=actor(o);b=mapping.action(actor(m));den=torch.linalg.vector_norm(a,dim=-1);errors.append((torch.linalg.vector_norm(a-b,dim=-1)/den.clamp_min(1e-8)).numpy());norms.append(den.numpy())
  e=np.concatenate(errors);den=np.concatenate(norms);assert np.isfinite(e).all()
  name=f"{record['group']}_s{record['training_seed']}";out=folder/(name+'.npz');np.savez_compressed(out,error=e,action_norm=den,source_training_seed=np.array(sources))
  results.append(dict(group=record['group'],training_seed=record['training_seed'],count=len(e),mean=float(e.mean(dtype=np.float64)),median=float(np.median(e)),p95=float(np.percentile(e,95)),small_denominator_count=int((den<1e-8).sum()),by_state_source_seed={str(i):float(e[np.array(sources)==i].mean(dtype=np.float64)) for i in range(1,6)},checkpoint_sha256=record['sha256'],output_sha256=sha(out)))
  print('EQUIVARIANCE_MEASURED',name,results[-1]['mean'],flush=True)
 (ROOT/'equivariance_statistics.json').write_text(json.dumps(dict(device='cpu',formula='per-state L2(pi(s)-Ma*pi(Ms(s)))/max(L2(pi(s)),1e-8); training reflection plus half-cycle',count=102400,source_manifest_sha256=sha(ROOT/'fixed_state_set.json'),mapping_contract_sha256=sha(contract_path),results=results,source_sha256=sha(__file__)),indent=2)+'\n')
if __name__=='__main__':main()
