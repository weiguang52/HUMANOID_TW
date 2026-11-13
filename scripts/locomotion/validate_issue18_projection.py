"""Validate the actual fifteen policies on all prespecified issue17 states."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from issue17_symmetry import TrainingMirror
from issue18_policy import StrictSymmetricPolicy
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
OLD=ROOT.with_name('issue17_20261113')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4)
 manifest=json.loads((OLD/'fixed_state_set.json').read_text());assert manifest['count']==102400
 states=[]
 for row in manifest['files']:
  assert sha(row['path'])==row['sha256']
  with np.load(row['path']) as z:states.append(z['actor'].reshape(-1,101))
 states=np.concatenate(states);assert states.shape==(102400,101)
 contract_path=Path('validation_artifacts/stage1_issue16_20261111/mirror/moments/control6000_s1_eval1/mirror_runtime_contract.json');contract=json.loads(contract_path.read_text());mapping=TrainingMirror(contract['joint_names'],contract['action_names'],contract['action_scales'])
 cohort=json.loads((OLD/'cohort.json').read_text());records=[dict(group='control6000',training_seed=i,**{k:cohort['models'][f'control6000_s{i}'][k] for k in ['checkpoint','sha256']}) for i in range(1,6)]+json.loads((OLD/'training_records.json').read_text());assert len(records)==15
 results=[];folder=ROOT/'projection_validation';folder.mkdir(exist_ok=True)
 for record in records:
  checkpoint=Path(record['checkpoint']);assert sha(checkpoint)==record['sha256'];dummy=TensorDict({'policy':torch.zeros(1,101)},batch_size=[1]);actor=MLPModel(dummy,{'actor':['policy']},'actor',30,hidden_dims=[512,256,128],activation='elu',obs_normalization=False,distribution_cfg={'class_name':'GaussianDistribution','init_std':.5,'std_type':'scalar'});loaded=torch.load(checkpoint,map_location='cpu',weights_only=False);actor.load_state_dict(loaded['actor_state_dict'],strict=True);actor.eval();before={k:v.clone() for k,v in actor.state_dict().items()};rng=torch.get_rng_state().clone();strict=StrictSymmetricPolicy(actor,mapping,'strict');identity=StrictSymmetricPolicy(actor,mapping,'identity');errors=[];absolute=[]
  with torch.inference_mode():
   for start in range(0,len(states),1024):
    x=torch.from_numpy(states[start:start+1024]);assert torch.equal(mapping.actor(mapping.actor(x)),x);o=TensorDict({'policy':x},batch_size=[len(x)]);t=TensorDict({'policy':mapping.actor(x)},batch_size=[len(x)]);original=actor(o);assert torch.equal(identity(o),original);a=strict(o);b=mapping.action(strict(t));assert torch.equal(mapping.action(mapping.action(a)),a);delta=torch.linalg.vector_norm(a-b,dim=-1);e=delta/torch.linalg.vector_norm(a,dim=-1).clamp_min(1e-8);assert torch.isfinite(e).all() and float(e.max())<=1e-5;errors.append(e.numpy());absolute.append(delta.numpy())
  assert torch.equal(rng,torch.get_rng_state());assert all(torch.equal(before[k],v) for k,v in actor.state_dict().items());assert sha(checkpoint)==record['sha256'];e=np.concatenate(errors);a=np.concatenate(absolute);name=record['group']+'_s'+str(record['training_seed']);out=folder/(name+'.npz');np.savez_compressed(out,relative_error=e,absolute_error=a);results.append(dict(model=name,checkpoint=str(checkpoint),checkpoint_sha256=record['sha256'],count=len(e),max_relative_error=float(e.max()),max_absolute_error=float(a.max()),identity_bitwise_equal_all_states=True,actor_and_optimizer_file_unchanged=True,inference_rng_unchanged=True,output_sha256=sha(out)));print('PROJECTION_FIXED_STATES_PASS',name,float(e.max()),flush=True)
 (ROOT/'projection_validation.json').write_text(json.dumps(dict(passed=True,models=15,states_per_model=102400,tolerance=1e-5,training_updates=0,results=results,fixed_state_manifest_sha256=sha(OLD/'fixed_state_set.json'),runtime_contract_sha256=sha(contract_path),source_sha256=sha(__file__),helper_source_sha256={n:sha(Path(__file__).with_name(n)) for n in ["issue18_policy.py","issue17_symmetry.py"]}),indent=2)+'\n')
if __name__=='__main__':main()
