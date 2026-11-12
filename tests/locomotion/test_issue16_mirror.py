"""Physical-axis, state, action and normalization tests for issue16 reflection."""
import json,math,sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/locomotion'))
from issue16_mirror import Mirror101,ReflectedPolicy,CONTRACT_PATH,counterpart

class TestReflection(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads(CONTRACT_PATH.read_text())
        self.names=[r['joint'] for r in self.contract['joints']]
        self.map=Mirror101(self.names)
    def test_all_thirty_axes_follow_axial_reflection(self):
        rows={r['joint']:r for r in self.contract['joints']}
        for name,row in rows.items():
            a=torch.tensor(row['axis_world_at_authored_pose'],dtype=torch.float64)
            b=torch.tensor(rows[counterpart(name)]['axis_world_at_authored_pose'],dtype=torch.float64)
            physical=torch.linalg.det(torch.diag(torch.tensor([1.,-1.,1.],dtype=torch.float64)))*b*torch.tensor([1.,-1.,1.])
            self.assertTrue(torch.allclose(a*row['sign'],physical,atol=1e-6,rtol=0),name)
        self.assertEqual(rows['left_hip_pitch_joint']['sign'],-1)
        self.assertEqual(rows['left_knee_pitch_joint']['sign'],1)
    def test_arbitrary_and_symmetric_state_involution(self):
        x=torch.arange(707,dtype=torch.float32).reshape(7,101)/37
        self.assertTrue(torch.equal(self.map.observation(self.map.observation(x)),x))
        symmetric=(x+self.map.observation(x))/2
        self.assertTrue(torch.equal(self.map.observation(symmetric),symmetric))
        self.assertTrue(torch.equal(self.map.action(self.map.action(x[:,:30])),x[:,:30]))
    def test_polar_axial_commands_and_no_input_mutation(self):
        x=torch.arange(101,dtype=torch.float32)[None];old=x.clone();y=self.map.observation(x)
        self.assertTrue(torch.equal(x,old))
        self.assertTrue(torch.equal(y[0,:9],x[0,:9]*torch.tensor([-1,1,-1,1,-1,1,1,-1,-1])))
    def test_global_gait_clock_invariant(self):
        phases=torch.arange(19,dtype=torch.float32)/19*2*math.pi
        x=torch.zeros(19,101);x[:,99]=phases.sin();x[:,100]=phases.cos()
        self.assertTrue(torch.equal(self.map.observation(x)[:,99:101],x[:,99:101]))
    def test_clock_driven_policy_true_spatial_conjugacy(self):
        class ClockPolicy:
            def __call__(self,obs):
                return obs['policy'][:,9:39]+obs['policy'][:,99:100]*torch.arange(1,31)
        policy=ClockPolicy();wrapped=ReflectedPolicy(policy,self.map,'mirror')
        x=torch.arange(202,dtype=torch.float32).reshape(2,101)/101
        physical_mirror=self.map.observation(x)
        physical_mirror[:,99:101]=x[:,99:101]  # Time does not reflect with spatial coordinates.
        actual=wrapped({'policy':physical_mirror})
        expected=self.map.action(policy({'policy':x}))
        self.assertTrue(torch.equal(actual,expected))
    def test_joint_blocks_and_distinct_action_order(self):
        order=list(reversed(self.names));m=Mirror101(self.names,order)
        x=torch.arange(101,dtype=torch.float32)[None];y=m.observation(x)
        for b in [9,39,69]:
            for i,name in enumerate(self.names):
                j=self.names.index(counterpart(name));s=self.contract['joints'][i]['sign']
                self.assertEqual(float(y[0,b+i]),s*float(x[0,b+j]))
        a=torch.arange(30,dtype=torch.float32)[None];out=m.action(a)
        for i,n in enumerate(order):
            sign=next(r['sign'] for r in self.contract['joints'] if r['joint']==n)
            self.assertEqual(float(out[0,i]),sign*float(a[0,order.index(counterpart(n))]))
    def test_absolute_desired_positions_and_scaled_action(self):
        scales=torch.linspace(.2,.6,30);m=Mirror101(self.names,action_scales=scales)
        d=torch.tensor([r['default'] for r in self.contract['joints']]);a=torch.arange(90,dtype=torch.float32).reshape(3,30)/50
        target=d+scales*a
        expected=target[:,m.perm]*m.sign
        actual=d+scales*m.action(a)
        self.assertTrue(torch.allclose(actual,expected,atol=3e-7,rtol=0))
        self.assertTrue(torch.allclose(m.action(m.action(a)),a,atol=3e-7,rtol=0))
    def test_reflection_precedes_normalization_and_preserves_critic(self):
        m=self.map
        class FakeModel:
            def __call__(self,obs):
                self.seen=obs
                return (obs['policy'][:,:30]-torch.arange(30))/torch.arange(1,31)
        model=FakeModel();wrapper=ReflectedPolicy(model,m,'mirror')
        x=torch.arange(202,dtype=torch.float32).reshape(2,101);c=torch.arange(268).reshape(2,134);obs={'policy':x,'critic':c}
        y=wrapper(obs);self.assertTrue(torch.equal(model.seen['policy'],m.observation(x)))
        self.assertTrue(torch.equal(model.seen['critic'],c));self.assertTrue(torch.equal(obs['policy'],x))
        expected=m.action((m.observation(x)[:,:30]-torch.arange(30))/torch.arange(1,31))
        self.assertTrue(torch.equal(y,expected))
    def test_mapping_and_wrapper_do_not_advance_rng(self):
        before=torch.get_rng_state().clone();m=Mirror101(self.names)
        x=torch.arange(202,dtype=torch.float32).reshape(2,101)
        m.observation(x);m.action(x[:,:30]);self.assertTrue(torch.equal(before,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
