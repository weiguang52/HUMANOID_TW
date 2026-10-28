"""Optional read-only training minibatch snapshots; never imported by default.

Returns the exact original batch objects. No extra environment/policy calls,
optimizer steps, random sampling or new training data. Caller selects a separate
new output directory and explicit batch ordinals before a diagnostic run.
"""
from pathlib import Path
import hashlib,inspect,json,os
import torch


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,tuple):return tuple(cpu_copy(x) for x in value)
    if value is None:return None
    raise ValueError('Only tensors, tensor tuples and None may be recorded')


class MinibatchCapture:
    def __init__(self,algorithm,directory,slots,provenance):
        self.algorithm=algorithm;self.original=algorithm.storage
        self.original_generator=algorithm.storage.mini_batch_generator
        self.directory=Path(directory)
        self.slots=tuple(slots)
        if not self.slots or len(self.slots)>16 or tuple(sorted(set(self.slots)))!=self.slots or any(type(x) is not int or x<0 for x in self.slots):
            raise ValueError('Expected1..16 distinct increasing nonnegative batch ordinals')
        self.provenance=json.loads(json.dumps(provenance,allow_nan=False))
        if algorithm.actor.is_recurrent or algorithm.critic.is_recurrent or algorithm.symmetry or algorithm.rnd or algorithm.is_multi_gpu:
            raise ValueError('Only audited feed-forward anchored PPO capture')
        if not hasattr(algorithm,'anchor_teacher'):
            raise ValueError('Expected explicit frozen anchor teacher')
        self.source={}
        for name,method in [('algorithm_update',type(algorithm).update),('algorithm_act',type(algorithm).act)]:
            code=inspect.getsource(method);self.source[name]={'sha256':hashlib.sha256(code.encode()).hexdigest(),'source':code}
        self.directory.mkdir(parents=True,exist_ok=False)
        self.yielded=0;self.records=[]

    def __getattr__(self,name):return getattr(self.original,name)

    def _capture(self,batch,index):
        alg=self.algorithm
        keys=['actions','old_actions_log_prob','values','advantages','returns','old_distribution_params','masks','hidden_states']
        payload={k:cpu_copy(getattr(batch,k)) for k in keys}
        payload['observations']={k:cpu_copy(v) for k,v in batch.observations.items()}
        payload['observation_batch_size']=list(batch.observations.batch_size)
        payload['actor_state_dict']={k:cpu_copy(v) for k,v in alg.actor.state_dict().items()}
        payload['critic_state_dict']={k:cpu_copy(v) for k,v in alg.critic.state_dict().items()}
        payload['teacher_state_dict']={k:cpu_copy(v) for k,v in alg.anchor_teacher.state_dict().items()}
        fields=['max_grad_norm','clip_param','value_loss_coef','entropy_coef','learning_rate','desired_kl','schedule','normalize_advantage_per_mini_batch','anchor_coefficient','anchor_temporal_coefficient','anchor_temporal_order','anchor_temporal_target','num_learning_epochs','num_mini_batches']
        payload['algorithm']={k:getattr(alg,k) for k in fields}
        payload['anchor_action_weights']=cpu_copy(alg.anchor_action_weights)
        payload['controlled_action_indices']=getattr(alg,'controlled_action_indices',None)
        payload['actor_model_repr']=str(alg.actor)
        payload['critic_model_repr']=str(alg.critic)
        payload['actor_distribution']={'class':type(alg.actor.distribution).__module__+'.'+type(alg.actor.distribution).__name__,'std_type':getattr(alg.actor.distribution,'std_type',None)}
        payload['actor_obs_groups']=list(alg.actor.obs_groups)
        payload['critic_obs_groups']=list(alg.critic.obs_groups)
        payload['batch_ordinal']=index;payload['provenance']=self.provenance
        payload['source']=self.source
        payload['position']='before original update consumes this batch; includes pre-normalization advantages'
        path=self.directory/f'batch_{index:04d}.pt';temporary=path.with_suffix('.partial')
        with temporary.open('xb') as stream:torch.save(payload,stream)
        try:os.link(temporary,path)  # Atomic publication without overwriting any existing file.
        finally:temporary.unlink()
        self.records.append({'ordinal':index,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples':int(batch.observations.batch_size[0])})

    def _capture_after(self,index):
        alg=self.algorithm
        payload={'batch_ordinal':index,'position':'after original backward, gradient clipping and optimizer step; before next batch',
                 'actor_gradients':{k:cpu_copy(v.grad) for k,v in alg.actor.named_parameters()},
                 'critic_gradients':{k:cpu_copy(v.grad) for k,v in alg.critic.named_parameters()},
                 'learning_rate':alg.learning_rate,
                 'optimizer_learning_rates':[g['lr'] for g in alg.optimizer.param_groups]}
        path=self.directory/f'batch_{index:04d}.after.pt';temporary=path.with_suffix('.partial')
        with temporary.open('xb') as stream:torch.save(payload,stream)
        try:os.link(temporary,path)
        finally:temporary.unlink()
        record=next(r for r in self.records if r['ordinal']==index)
        record.update(after_path=str(path),after_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def mini_batch_generator(self,*args,**kwargs):
        for batch in self.original_generator(*args,**kwargs):
            index=self.yielded
            if index in self.slots:self._capture(batch,index)
            self.yielded+=1
            yield batch
            if index in self.slots:self._capture_after(index)

    def finish(self):
        captured=tuple(r['ordinal'] for r in self.records)
        if captured!=self.slots or any('after_path' not in r for r in self.records):raise RuntimeError('Diagnostic run ended before all declared before/after batches were captured')
        report={'complete':True,'requested_ordinals':list(self.slots),'yielded_batches':self.yielded,'records':self.records,'provenance':self.provenance,'source_sha256':{k:v['sha256'] for k,v in self.source.items()},'scope':'Read-only actual training minibatch capture, not a training improvement or causal conclusion'}
        path=self.directory/'manifest.json'
        with path.open('x') as stream:json.dump(report,stream,indent=2)
        return report


def install_capture(algorithm,directory,slots=(0,19),provenance=None):
    if getattr(algorithm.storage,'_training_minibatch_capture',None) is not None:raise ValueError('Capture already installed')
    capture=MinibatchCapture(algorithm,directory,slots,{} if provenance is None else provenance)
    algorithm.storage.mini_batch_generator=capture.mini_batch_generator
    algorithm.storage._training_minibatch_capture=capture
    return capture
