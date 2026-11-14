"""Check every final issue19 model and export all five strict actors."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch,onnx,onnxruntime as ort
from tensordict import TensorDict
from validate_issue19_actor import actor
from issue17_symmetry import TrainingMirror
from issue19_actor import install_actor,ExportActor
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4)
 assert (ROOT/'formal_training_complete').exists()
 records=json.loads((ROOT/'training_records.json').read_text());assert len(records)==10
 manifest=json.loads((ROOT.with_name('issue17_20261113')/'fixed_state_set.json').read_text());xs=[]
 for row in manifest['files']:
  assert sha(row['path'])==row['sha256']
  with np.load(row['path']) as z:xs.append(z['actor'].reshape(-1,101))
 states=np.concatenate(xs);assert states.shape==(102400,101)
 c=json.loads(Path('validation_artifacts/stage1_issue16_20261111/mirror/moments/control6000_s1_eval1/mirror_runtime_contract.json').read_text())
 m=TrainingMirror(c['joint_names'],c['action_names'],c['action_scales'])
 out=ROOT/'onnx';out.mkdir(exist_ok=True);results=[]
 for r in records:
  assert sha(r['checkpoint'])==r['sha256']
  saved=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
  net=actor()
  if r['group']=='sym_wrap':install_actor(net,m)
  net.load_state_dict(saved['actor_state_dict'],strict=True);net.eval()
  assert saved['iter']==14992
  assert all(torch.isfinite(t).all() for t in net.state_dict().values())
  if r['group']!='sym_wrap':continue
  d=net.distribution;std=d.shared_std_param.detach()[d.orbit_index];assert torch.equal(std,std[m.spatial.action_perm])
  path=out/('sym_wrap_s'+str(r['training_seed'])+'.onnx')
  exported=ExportActor(net).eval()
  torch.onnx.export(exported,(torch.zeros(1,101),),str(path),opset_version=18,input_names=['obs'],output_names=['actions'],dynamic_axes={'obs':{0:'batch'},'actions':{0:'batch'}},dynamo=False)
  graph=onnx.load(str(path));onnx.checker.check_model(graph);assert sum(n.op_type=='Gemm' for n in graph.graph.node)==8
  options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
  session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
  maximum=0.;onnx_error=0.
  with torch.inference_mode():
   for start in range(0,len(states),1024):
    x=torch.from_numpy(states[start:start+1024]);obs=TensorDict({'policy':x},[len(x)])
    y=net(obs);reflected=net(TensorDict({'policy':m.actor(x)},[len(x)]))
    maximum=max(maximum,float((reflected-m.action(y)).abs().max()))
    onnx_error=max(onnx_error,float(np.max(np.abs(session.run(None,{'obs':x.numpy()})[0]-y.numpy()))))
  assert maximum<=1e-5 and onnx_error<=1e-5
  results.append(dict(seed=r['training_seed'],checkpoint=r['checkpoint'],sha256=r['sha256'],onnx=str(path),onnx_sha256=sha(path),max_equivariance_error=maximum,onnx_max_error=onnx_error,shared_std=std.tolist()))
  print('FINAL_SYMMETRIC_ACTOR_ONNX_PASS',r['training_seed'],maximum,onnx_error,flush=True)
 assert len(results)==5
 (ROOT/'final_model_validation.json').write_text(json.dumps(dict(passed=True,all_final_checkpoints_verified=10,wrapped_models=5,states_per_model=102400,results=results),indent=2))
if __name__=='__main__':main()
