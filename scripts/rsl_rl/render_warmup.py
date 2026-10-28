"""Opt-in renderer warm-up, rejecting physics/command/RNG changes."""
import random
import numpy as np
import torch

def state_snapshot(base):
    robot=base.scene['robot'];command=base.command_manager.get_term('motion')
    tensors={'joint_pos':robot.data.joint_pos,'joint_vel':robot.data.joint_vel,
        'root_state':robot.data.root_state_w,'motion_steps':command.time_steps,
        'motion_ids':command.motion_ids,'action':base.action_manager.action,
        'previous_action':base.action_manager.prev_action}
    counters={'physics_steps':base._sim_step_counter,'control_steps':base.common_step_counter,
        'simulation_time':base.sim.current_time}
    return {k:v.detach().clone() for k,v in tensors.items()},dict(counters)

def numpy_rng_equal(a,b):return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]

@torch.inference_mode()
def warm_up_renderer(base,frames=0):
    if type(frames) is not int or not 0<=frames<=64:raise ValueError('Warm-up frames must be an integer0..64')
    if frames==0:return None
    initial,counters=state_snapshot(base)
    py_rng=random.getstate();np_rng=np.random.get_state();torch_rng=torch.get_rng_state().clone()
    device=torch.device(base.device);cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    observations=[]
    for _ in range(frames):
        base.sim.render()
        image=np.asarray(base.render(recompute=True))
        if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:
            raise RuntimeError('Expected actual uint8 RGB renderer output')
        observations.append({'max':int(image.max()),'std':float(image.std()),'mean':float(image.mean())})
    final,after=state_snapshot(base)
    changed=[k for k in initial if not torch.equal(initial[k],final[k])]
    changed.extend(k for k in counters if counters[k]!=after[k])
    if changed:raise RuntimeError('Render warm-up changed physics/command state: '+','.join(changed))
    rng_unchanged=random.getstate()==py_rng and numpy_rng_equal(np_rng,np.random.get_state()) and torch.equal(torch_rng,torch.get_rng_state())
    if cuda_rng is not None:rng_unchanged=rng_unchanged and torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
    if not rng_unchanged:raise RuntimeError('Render warm-up consumed policy/shared RNG')
    if observations[-1]['max']<=2 or observations[-1]['std']<.5:
        raise RuntimeError('Renderer still empty/uniform after warm-up')
    return {'frames':frames,'operation':'sim.render then RGB read; no env.step',
        'physics_and_command_unchanged':True,'RNG_unchanged':True,
        'snapshot_tensors':list(initial),'snapshot_counters':list(counters),
        'render_observations':observations,'scope':'Only render initialization, not motion quality or complete visual acceptance'}
