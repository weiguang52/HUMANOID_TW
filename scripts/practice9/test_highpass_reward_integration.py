from pathlib import Path
from types import SimpleNamespace as NS,ModuleType
import sys,importlib.util,json,copy,torch,numpy as np,ast
ROOT=Path(__file__).resolve().parents[2]
MDP=ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp'
sys.path.insert(0,str(ROOT/'scripts/rsl_rl'))
from highpass_reward_contract import contract,validate_resume,JOINTS
package=ModuleType('hp_test');package.__path__=[str(MDP)];sys.modules['hp_test']=package

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
new=load('hp_test.leg_error_highpass',MDP/'leg_error_highpass.py')
old=load('hp_legacy',Path('/root/gpufree-data/datasets/practice9/highpass_leg_slip_20261025/source_snapshot/leg_error_highpass.py'))

def fake():
 names=JOINTS[::-1]
 cmd=NS(robot=NS(joint_names=names),joint_indexes=torch.arange(12),robot_joint_pos=torch.zeros(3,12),joint_pos=torch.zeros(3,12),cfg=NS(body_names=['base_link']),body_lin_vel_w=torch.tensor([[[0.,0.,0.]],[[.075,0.,0.]],[[.1,0.,0.]]]))
 return NS(command_manager=NS(get_term=lambda _:cmd),step_dt=.02,episode_length_buf=torch.full((3,),3)),cmd

a,ac=fake();b,bc=fake();rng=torch.Generator().manual_seed(20261026)
for step in range(512):
 data=torch.randn(3,12,generator=rng)*.1
 reset=torch.tensor([step%47==0,step%29==0,step%91==0])
 for env,cmd in [(a,ac),(b,bc)]:env.episode_length_buf=torch.where(reset,0,3);cmd.robot_joint_pos=data.clone()
 assert torch.equal(new.moving_leg_highpass_error(a),old.moving_leg_highpass_error(b))
a,cmd=fake();expected_filter=load('hp_test.sos_error_highpass',MDP/'sos_error_highpass.py').ButterworthErrorHighpass(8.,.02)
for step in range(256):
 cmd.robot_joint_pos=torch.randn(3,12,generator=rng)*.1
 a.episode_length_buf=torch.tensor([0 if step%43==0 else 3,0 if step%19==0 else 3,3])
 raw=cmd.robot_joint_pos[:,list(range(11,-1,-1))]
 high=expected_filter.update(raw,a.episode_length_buf<=1)
 cost=new.moving_leg_highpass_error(a,filter_order=4,trace_steps=16)
 gate=torch.tensor([0.,.5,1.]);expected=gate*(high/.01).square().mean(-1)
 assert torch.allclose(cost,expected,atol=2e-5,rtol=1e-5)
 assert cost[0]==0
assert len(cmd._leg_error_highpass_trace)==16 and cmd._leg_error_highpass_updates==256
first=cmd._leg_error_highpass_trace[0]['error'].copy();cmd.robot_joint_pos.fill_(99)
assert np.array_equal(first,cmd._leg_error_highpass_trace[0]['error'])
try:new.moving_leg_highpass_error(a,filter_order=1)
except ValueError:pass
else:raise AssertionError('Changed live order accepted')
for order in [2,True,4.0]:
 env,_=fake()
 try:new.moving_leg_highpass_error(env,filter_order=order)
 except ValueError:pass
 else:raise AssertionError('Unsupported order accepted')
cfg=NS(rewards=NS(moving_leg_highpass=NS(weight=-.2,params={'filter_order':4,'cutoff_hz':8.,'scale':.01})),sim=NS(dt=.005),decimation=4)
saved=contract(cfg);validate_resume(json.loads(json.dumps(saved)),contract(cfg))
for field,value in [('filter_order',1),('scale',.02),('cutoff_hz',7.)]:
 bad=copy.deepcopy(cfg);bad.rewards.moving_leg_highpass.params[field]=value
 try:validate_resume(saved,contract(bad))
 except ValueError:pass
 else:raise AssertionError('Changed filter contract accepted')
for requested in [None,dict(saved,weight=-.3)]:
 try:validate_resume(saved,requested)
 except ValueError:pass
 else:raise AssertionError('Removed or changed reward accepted')
validate_resume(None,saved)
proto=Path('/root/gpufree-data/datasets/practice9/sos_highpass_20261026/sos_highpass.py')
classes=[next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.ClassDef)) for p in [proto,MDP/'sos_error_highpass.py']]
assert ast.dump(classes[0])==ast.dump(classes[1])
print('PASS legacy reward512 steps bitwise, SOS reward256 steps, gating, async resets, canonical joint order, immutable trace, runtime order lock, resume semantic rejection, tested prototype class AST match')
