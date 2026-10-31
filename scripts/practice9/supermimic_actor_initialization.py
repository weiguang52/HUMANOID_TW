"""Explicit pilot initialization, never applied implicitly or during inference."""
import torch

def zero_readout(actor, optimizer):
    readout = actor.decoder[-1]
    if not isinstance(readout, torch.nn.Linear):
        raise ValueError('Expected final linear action readout')
    prefix = f'decoder.{len(actor.decoder)-1}.'
    before = {k:v.detach().clone() for k,v in actor.state_dict().items()}
    removed = 0
    with torch.no_grad():
        for parameter in readout.parameters():
            parameter.zero_()
            removed += int(parameter in optimizer.state)
            optimizer.state.pop(parameter, None)
    for name,value in actor.state_dict().items():
        if name.startswith(prefix):
            if not bool((value==0).all()):raise AssertionError(name)
        elif not torch.equal(value,before[name]):raise AssertionError(name)
    return dict(readout_prefix=prefix,only_readout_zeroed=True,other_actor_critic_tensors_unchanged=True,removed_optimizer_entries=removed,inference_mask_or_clip=False)
