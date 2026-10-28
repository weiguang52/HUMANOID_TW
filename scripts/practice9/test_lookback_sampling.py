from pathlib import Path
import importlib.util,torch,sys,tempfile,json
root=Path(__file__).resolve().parents[2]
p=root/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/lookback_sampling.py'
spec=importlib.util.spec_from_file_location('lookback',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);preceding_failure_score=m.preceding_failure_score
def test():
 rows=[]
 for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
  score=torch.tensor([1.,.8,1.,.6],device=device);visits=torch.tensor([0.,4.,0.,2.],device=device);ids=torch.tensor([0,0,1,1],device=device);starts=torch.tensor([0.,50.,0.,50.],device=device)
  out=preceding_failure_score(score,visits,ids,starts);torch.testing.assert_close(out,torch.tensor([1.4,.4,1.3,.3],device=device));torch.testing.assert_close(out.sum(),score.sum())
  assert torch.equal(preceding_failure_score(score,visits,ids,starts,0),score)
  boundary=torch.tensor([.7,0.,.9,0.],device=device);assert torch.equal(preceding_failure_score(boundary,torch.ones_like(visits),ids,starts),boundary)
  assert torch.equal(preceding_failure_score(score,torch.zeros_like(visits),ids,starts),score)
  prior=torch.tensor([.1,.2,.3,.4],device=device);p=.8*prior+.2*out/out.sum();assert torch.all(p>=.8*prior-1e-7);torch.testing.assert_close(p.sum(),torch.tensor(1.,device=device));rows.append({'device':device,'passed':True,'score':out.cpu().tolist()})
 return rows

print(test())
sys.path.insert(0,str(root/'scripts/rsl_rl'))
from lookback_contract import prepare
from types import SimpleNamespace as NS
cfg=NS(commands=NS(motion=NS(adaptive_lookback_share=.5,adaptive_uniform_ratio=.8,adaptive_bin_size_s=1.,adaptive_kernel_size=1)))
with tempfile.TemporaryDirectory() as t:
 d=Path(t);a=prepare(cfg,d);(d/'model.pt').touch();assert prepare(cfg,d,d/'model.pt')==a
 cfg.commands.motion.adaptive_lookback_share=0.
 try:prepare(cfg,d,d/'model.pt')
 except ValueError:pass
 else:raise AssertionError('silent resume change')
print('resume contract rejection PASS')
