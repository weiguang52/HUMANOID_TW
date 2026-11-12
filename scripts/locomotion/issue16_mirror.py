"""Inference-only sagittal reflection for the exact TW101 actor observation contract."""
import json
from pathlib import Path
import torch

CONTRACT_PATH=Path(__file__).resolve().parents[2]/'tests/locomotion/fixtures/issue16_joint_axes.json'
TERMS=['base_ang_vel','projected_gravity','velocity_commands','joint_pos_rel','joint_vel_rel','last_action','gait_phase']

def counterpart(name):
    if name.startswith('left_'):return 'right_'+name[5:]
    if name.startswith('right_'):return 'left_'+name[6:]
    return name

class Mirror101:
    def __init__(self,joint_names,action_names=None,action_scales=None,contract=None,device='cpu'):
        contract=contract or json.loads(CONTRACT_PATH.read_text())
        rows={r['joint']:r for r in contract['joints']}
        self.joint_names=list(joint_names);self.action_names=list(action_names or joint_names)
        assert len(self.joint_names)==len(self.action_names)==30
        assert set(rows)==set(self.joint_names)==set(self.action_names)
        for name,row in rows.items():
            assert row['partner']==counterpart(name) and row['sign'] in [-1,1]
            assert rows[row['partner']]['partner']==name and rows[row['partner']]['sign']==row['sign']
            a=torch.tensor(row['axis_world_at_authored_pose'],dtype=torch.float64)
            b=torch.tensor(rows[row['partner']]['axis_world_at_authored_pose'],dtype=torch.float64)*torch.tensor([-1.,1.,-1.],dtype=torch.float64)
            assert torch.linalg.vector_norm(a-row['sign']*b)<1e-6,(name,a,b)
        self.perm=torch.tensor([self.joint_names.index(counterpart(n)) for n in self.joint_names],device=device)
        self.sign=torch.tensor([rows[n]['sign'] for n in self.joint_names],device=device,dtype=torch.float32)
        self.action_perm=torch.tensor([self.action_names.index(counterpart(n)) for n in self.action_names],device=device)
        signs=torch.tensor([rows[n]['sign'] for n in self.action_names],device=device,dtype=torch.float32)
        scales=torch.ones(30,device=device) if action_scales is None else torch.as_tensor(action_scales,device=device,dtype=torch.float32)
        assert scales.shape==(30,) and torch.all(scales>0)
        self.action_factor=signs*scales[self.action_perm]/scales
        self.contract=contract

    def observation(self,x):
        assert x.shape[-1]==101
        y=x.clone()
        y[...,:3]=x[...,:3]*x.new_tensor([-1,1,-1])
        y[...,3:6]=x[...,3:6]*x.new_tensor([1,-1,1])
        y[...,6:9]=x[...,6:9]*x.new_tensor([1,-1,-1])
        for start in [9,39,69]:y[...,start:start+30]=x[...,start:start+30].index_select(-1,self.perm)*self.sign
        y[...,99:101]=x[...,99:101]
        return y

    def action(self,x):
        assert x.shape[-1]==30
        return x.index_select(-1,self.action_perm)*self.action_factor

    def observation_dict(self,obs):
        result=obs.clone() if hasattr(obs,'clone') else dict(obs)
        result['policy']=self.observation(obs['policy'])
        return result

class ReflectedPolicy:
    def __init__(self,policy,mapping,mode,base=None):
        assert mode in ['original','identity','mirror','default']
        self.policy,self.mapping,self.mode,self.base=policy,mapping,mode,base
        self.calls=0
    def __call__(self,obs):
        self.calls+=1
        if self.mode=='default':
            return obs['policy'].new_zeros((obs['policy'].shape[0],30))
        if self.mode=='original':return self.policy(obs)
        if self.mode=='identity':
            identity=obs.clone() if hasattr(obs,'clone') else dict(obs)
            return self.policy(identity)
        # The original MLP performs normalization internally; reflect RAW actor features first.
        return self.mapping.action(self.policy(self.mapping.observation_dict(obs)))

def install_policy(policy,base,mode,output):
    robot=base.scene['robot'];term=base.action_manager.get_term('JointPositionAction')
    terms=base.observation_manager.active_terms['policy']
    assert terms==TERMS,(terms,TERMS)
    dims=base.observation_manager.group_obs_term_dim['policy']
    assert [int(torch.tensor(x).prod()) for x in dims]==[3,3,3,30,30,30,2],dims
    ids=term._joint_ids
    ids=list(range(len(robot.joint_names)))[ids] if isinstance(ids,slice) else list(ids)
    ids=[int(x) for x in ids];action_names=[robot.joint_names[i] for i in ids]
    scale=term._scale
    if isinstance(scale,(int,float)):scale=torch.full((30,),float(scale),device=base.device)
    else:scale=scale[0] if scale.ndim==2 else scale
    mapping=Mirror101(robot.joint_names,action_names,scale,device=base.device)
    defaults=robot.data.default_joint_pos
    assert torch.equal(defaults,defaults[:1].expand_as(defaults))
    assert torch.allclose(defaults[0],defaults[0][mapping.perm]*mapping.sign,atol=1e-6,rtol=0),'Defaults are not physically reflected'
    offset=term._offset
    if isinstance(offset,torch.Tensor):
        assert torch.allclose(offset,defaults[:,ids],atol=1e-7,rtol=0),'Unexpected action offset'
    else:raise AssertionError('Expected default joint-position action offset')
    groups=getattr(policy,'obs_groups',None)
    assert list(groups)==['policy'],groups
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metadata={'mode':mode,'training_updates':0,'joint_names':robot.joint_names,'action_names':action_names,
        'action_scales':scale.detach().cpu().tolist(),'defaults':defaults[0].detach().cpu().tolist(),
        'terms':terms,'dimensions':[list(x) for x in dims],'actor_dim':101,
        'mirror_applied_before_existing_MLP_normalizer':True,'critic_changed':False,
        'gait_reflection':'Global scalar clock (sin,cos) unchanged by pure spatial reflection; no time/phase intervention',
        'absolute_joint_reflection':'q_i -> sign_i*q_partner; zero affine offsets verified at actual defaults',
        'action_reflection':'a_i=sign_i*(scale_partner/scale_i)*a_partner; actual default offsets verified',
        'usd_sha256':mapping.contract['usd_sha256']}
    (output/'mirror_runtime_contract.json').write_text(json.dumps(metadata,indent=2)+'\n')
    wrapped=ReflectedPolicy(policy,mapping,mode,base)
    return wrapped,metadata
