"""Check phase+10 versus pure-spatial reflected policy on all fixed states."""
import json,numpy as np,torch
from pathlib import Path
from tensordict import TensorDict
from validate_issue19_actor import actor
from issue17_symmetry import TrainingMirror
from issue19_actor import install_actor
from issue20_common import ROOT,OLD,REPO,sha,save

def main():
 torch.set_num_threads(4)
 manifest=json.loads((OLD.with_name('issue17_20261113')/'fixed_state_set.json').read_text());xs=[]
 for row in manifest['files']:
  assert sha(row['path'])==row['sha256']
  with np.load(row['path']) as z:xs.append(z['actor'].reshape(-1,101))
 states=np.concatenate(xs);assert states.shape==(102400,101)
 c=json.loads((REPO/'validation_artifacts/stage1_issue16_20261111/mirror/moments/control6000_s1_eval1/mirror_runtime_contract.json').read_text());m=TrainingMirror(c['joint_names'],c['action_names'],c['action_scales'])
 models=[r for r in json.loads((OLD/'training_records.json').read_text()) if r['group']=='sym_wrap'];assert len(models)==5
 save('cohort.json',models);results=[]
 for row in models:
  assert sha(row['checkpoint'])==row['sha256'];net=actor();install_actor(net,m);net.load_state_dict(torch.load(row['checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict']);net.eval();maximum=0.;clock_max=0.
  with torch.inference_mode():
   for start in range(0,len(states),1024):
    x=torch.from_numpy(states[start:start+1024]);phase=x.clone();phase[:,99:101]=-phase[:,99:101]
    left=net(TensorDict({'policy':phase},[len(x)]));right=m.action(net(TensorDict({'policy':m.spatial.observation(x)},[len(x)])));maximum=max(maximum,float((left-right).abs().max()))
    # Also exercise actual float32 idx arithmetic, rather than assuming sin(x+pi)=-sin(x) bitwise.
    idx=torch.arange(len(x))%20;xp=x.clone();xh=x.clone();q=idx.float()/20;qh=((idx+10)%20).float()/20
    xp[:,99]=torch.sin(q*torch.pi*2);xp[:,100]=torch.cos(q*torch.pi*2);xh[:,99]=torch.sin(qh*torch.pi*2);xh[:,100]=torch.cos(qh*torch.pi*2)
    a=net(TensorDict({'policy':xh},[len(x)]));b=m.action(net(TensorDict({'policy':m.spatial.observation(xp)},[len(x)])));clock_max=max(clock_max,float((a-b).abs().max()))
  assert maximum<=1e-5 and clock_max<=1e-5,(maximum,clock_max)
  results.append({'seed':row['training_seed'],'checkpoint_sha256':row['sha256'],'states':len(states),'algebraic_phase_max_error':maximum,'actual_idx_float32_phase_max_error':clock_max});print('ACTOR_PHASE_PASS',results[-1],flush=True)
 save('actor_phase_validation.json',{'passed':True,'results':results,'rules_sha256':sha(ROOT/'classification_rules.json'),'pure_spatial_M_gait_unchanged':True,'training_updates':0})
if __name__=='__main__':main()
