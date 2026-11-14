"""Pretraining gates for issue19; actual parent checkpoints and frozen states."""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from issue17_symmetry import TrainingMirror
from issue18_policy import StrictSymmetricPolicy
from issue19_actor import install_actor, ExportActor

ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
OLD=ROOT.with_name('issue17_20261113')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actor():
    return MLPModel(TensorDict({'policy':torch.zeros(1,101)},[1]),
        {'actor':['policy']},'actor',30,hidden_dims=[512,256,128],
        activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.5,'std_type':'scalar'})
def main():
    import onnx, onnxruntime as ort
    torch.set_num_threads(4)
    manifest=json.loads((OLD/'fixed_state_set.json').read_text())
    xs=[]
    for row in manifest['files']:
        assert sha(row['path'])==row['sha256']
        with np.load(row['path']) as z: xs.append(z['actor'].reshape(-1,101))
    states=np.concatenate(xs);assert states.shape==(102400,101)
    c=json.loads(Path('validation_artifacts/stage1_issue16_20261111/mirror/moments/control6000_s1_eval1/mirror_runtime_contract.json').read_text())
    m=TrainingMirror(c['joint_names'],c['action_names'],c['action_scales'])
    records=[r for r in json.loads((OLD/'training_records.json').read_text()) if r['group']=='sym_int']
    results=[];out=ROOT/'pretraining';out.mkdir(exist_ok=True)
    for r in records:
        assert sha(r['checkpoint'])==r['sha256']
        loaded=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        base=actor();base.load_state_dict(loaded['actor_state_dict'])
        new=copy.deepcopy(base)
        # Load the exact optimizer slot for each actor parameter; critic slots tested at runner smoke.
        opt=torch.optim.Adam(new.parameters())
        parent_opt=loaded['optimizer_state_dict']; params=parent_opt['param_groups'][0]['params']
        for p,idx in zip(new.parameters(),params):
            opt.state[p]=copy.deepcopy(parent_opt['state'][idx])
        other={id(p):copy.deepcopy(opt.state[p]) for p in new.parameters() if p is not new.distribution.std_param}
        audit=install_actor(new,m,opt)
        for p in new.parameters():
            if id(p) in other:
                for k,v in other[id(p)].items():
                    assert torch.equal(v,opt.state[p][k]) if isinstance(v,torch.Tensor) else v==opt.state[p][k]
        strict=StrictSymmetricPolicy(base,m,'strict')
        maximum=0.
        with torch.inference_mode():
            for start in range(0,len(states),1024):
                x=torch.from_numpy(states[start:start+1024]);obs=TensorDict({'policy':x},[len(x)])
                y=new(obs);assert torch.equal(y,strict(obs))
                err=float((new(TensorDict({'policy':m.actor(x)},[len(x)]))-m.action(y)).abs().max())
                maximum=max(maximum,err);assert err<=1e-5
        # The mirrored probability, entropy, and ratio refer to the executed projected mean.
        x=torch.from_numpy(states[:128]);o=TensorDict({'policy':x},[128]);t=TensorDict({'policy':m.actor(x)},[128])
        action=new(o,stochastic_output=True);lp=new.get_output_log_prob(action);entropy=new.output_entropy
        assert torch.equal(new.output_mean,new(o))
        direct=torch.distributions.Normal(new.output_mean,new.output_std)
        assert torch.equal(lp,direct.log_prob(action).sum(-1))
        assert torch.equal(entropy,direct.entropy().sum(-1))
        new(t,stochastic_output=True);mlp=new.get_output_log_prob(m.action(action))
        assert torch.allclose(lp,mlp,atol=2e-4,rtol=1e-5),(lp-mlp).abs().max()
        loss=-mlp.mean();opt.zero_grad();loss.backward()
        assert new.distribution.shared_std_param.grad is not None
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.mlp.parameters())
        export=ExportActor(new).eval();path=out/('sym_wrap_s'+str(r['training_seed'])+'_initial.onnx')
        torch.onnx.export(export,(torch.zeros(1,101),),str(path),opset_version=18,input_names=['obs'],output_names=['actions'],dynamic_axes={'obs':{0:'batch'},'actions':{0:'batch'}},dynamo=False)
        graph=onnx.load(str(path));onnx.checker.check_model(graph)
        assert len([node for node in graph.graph.node if node.op_type=='Gemm'])==8
        session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);oe=0.
        with torch.inference_mode():
            for start in range(0,len(states),1024):
                x=states[start:start+1024]
                a=session.run(None,{'obs':x})[0]
                b=new(TensorDict({'policy':torch.from_numpy(x)},[len(x)])).numpy()
                oe=max(oe,float(np.max(np.abs(a-b))))
        assert oe<=1e-5,oe
        results.append(dict(seed=r['training_seed'],checkpoint=r['checkpoint'],parent_sha256=r['sha256'],states=102400,max_equivariance_error=maximum,initial_strict_output_bitwise_equal=True,onnx_max_absolute_error=oe,onnx_sha256=sha(path),onnx_gemm_nodes=8,std=audit,probability_entropy_and_gradient_checks=True))
        print('ISSUE19_PRETRAINING_PASS',r['training_seed'],maximum,oe,flush=True)
    (ROOT/'actor_pretraining_validation.json').write_text(json.dumps(dict(passed=True,models=5,results=results,rules_sha256=sha(ROOT/'classification_rules.json')),indent=2))
if __name__=='__main__': main()
