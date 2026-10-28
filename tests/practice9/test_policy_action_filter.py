"""Episode isolation and name-order checks for the opt-in replay filter."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/rsl_rl'))
from policy_action_filter import LegActionFilter

def test_reset_and_permuted_order():
    names=['right_ankle_pitch_joint','waist_yaw','left_ankle_yaw_joint','left_elbow_pitch_joint','right_ankle_yaw_joint','left_ankle_pitch_joint']
    f=LegActionFilter(names,'cpu',.5,'ankles')
    a=torch.arange(6,dtype=torch.float32)[None,:];b=a+2;c=a+6
    rng=torch.random.get_rng_state().clone()
    assert f(a) is a
    out=f(b)
    torch.testing.assert_close(out[:,f.mask],a[:,f.mask]+1,rtol=0,atol=0)
    torch.testing.assert_close(out[:,~f.mask],b[:,~f.mask],rtol=0,atol=0)
    f.reset(torch.tensor([False]))
    out=f(c)
    torch.testing.assert_close(out[:,f.mask],a[:,f.mask]+3.5,rtol=0,atol=0)
    f.reset(torch.tensor([True]))
    assert f(c) is c
    fresh=LegActionFilter(names,'cpu',.5,'ankles')
    torch.testing.assert_close(f(b), (c+b)*.5* f.mask + b*(~f.mask),rtol=0,atol=0)
    assert fresh(b) is b
    assert torch.equal(rng,torch.random.get_rng_state())
    torch.testing.assert_close(a,torch.arange(6,dtype=torch.float32)[None,:],rtol=0,atol=0)

if __name__=='__main__':
    test_reset_and_permuted_order()
    print('PASS: name order, recurrence, nonleg identity, reset isolation, input/RNG preservation')

def test_reference_residual():
    names=['left_ankle_yaw_joint','left_ankle_pitch_joint','right_ankle_yaw_joint','right_ankle_pitch_joint','waist_yaw']
    absolute=LegActionFilter(names,'cpu',.5,'ankles')
    residual=LegActionFilter(names,'cpu',.5,'ankles')
    ref=torch.full((1,5),2.)
    for x in [torch.zeros(1,5),torch.ones(1,5),torch.full((1,5),4.)]:
        torch.testing.assert_close(absolute(x),residual(x,ref),rtol=0,atol=0)
    residual.reset(torch.tensor([True]))
    for value in [0.,4.,-2.,6.]:
        moving=torch.full((1,5),value)
        torch.testing.assert_close(residual(moving,moving),moving,rtol=0,atol=0)
    residual.reset(torch.tensor([True]))
    action=torch.arange(5,dtype=torch.float32)[None,:]
    assert residual(action,torch.zeros_like(action)) is action
    out=residual(action+2,torch.ones_like(action))
    torch.testing.assert_close(out[:,~residual.mask],(action+2)[:,~residual.mask],rtol=0,atol=0)

if __name__=='__main__':
    test_reference_residual()
    print('PASS: constant-reference parity, moving-reference feedforward, reset and unfiltered-joint preservation')
