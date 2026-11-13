"""Bitwise comparison of two actual training update paths, including optimizer/RNG/physics."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def equal(a,b,path='root'):
 assert type(a)==type(b),(path,type(a),type(b))
 if isinstance(a,torch.Tensor):
  assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),path
 elif isinstance(a,np.ndarray):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),path
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:equal(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
 else:assert a==b,(path,a,b)
def main():
 assert (ROOT/'preflight_runs_complete').exists()
 bindings={}
 for name in ['final.pt','runtime.pt']:
  paths=[ROOT/'preflight'/m/name for m in ['original','disabled']]
  values=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
  equal(*values,path=name)
  for p in paths:bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  print('BITWISE_IDENTICAL',name,list(values[0]),flush=True)
 (ROOT/'disabled_two_update_reproduction.json').write_text(json.dumps(dict(passed=True,num_envs=4096,updates_per_run=2,original_and_disabled_model_optimizer_std_normalizers_equal=True,all_saved_CPU_CUDA_Python_NumPy_RNG_root_joint_states_equal=True,inputs=bindings),indent=2)+'\n')
if __name__=='__main__':main()
