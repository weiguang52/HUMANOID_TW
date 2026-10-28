"""Opt-in immutable CPU snapshots for same-observation policy comparison."""
import numpy as np

class PolicyObservationTrace:
    def __init__(self,env=None):
        self.observations=[]
        self.actions=[]
        self.keys=None
        self.base_schema=None
        if env is not None:
            manager=env.unwrapped.observation_manager
            names=list(manager.active_terms['policy'])
            shapes=manager.group_obs_term_dim['policy']
            if len(names)!=len(shapes) or len(set(names))!=len(names):
                raise ValueError('Invalid live observation term schema')
            bounds=[];offset=0
            for shape in shapes:
                width=int(np.prod(shape))
                if width<=0:raise ValueError('Empty observation term')
                bounds.append((offset,offset+width));offset+=width
            if offset!=int(np.prod(manager.group_obs_dim['policy'])):
                raise ValueError('Concatenated observation schema dimension differs')
            self.base_schema=(names,bounds,offset)

    def append(self, observations, actions):
        keys=tuple(sorted(observations.keys()))
        if not all(isinstance(k,str) for k in keys):
            raise ValueError('Only flat named observation tensors are supported')
        if self.keys is not None and keys!=self.keys:
            raise ValueError('Observation keys changed during replay')
        if self.base_schema is not None:
            base=observations.get('contact_base_policy',observations['policy'])
            if base.shape[-1]!=self.base_schema[2]:raise ValueError('Recorded base policy dimension differs')
        self.keys=keys
        self.observations.append({k:observations[k].detach().cpu().numpy().copy() for k in keys})
        self.actions.append(actions.detach().cpu().numpy().copy())

    def save(self,path):
        if not self.actions:
            raise ValueError('Cannot save empty policy trace')
        arrays={'action':np.stack(self.actions),'observation_keys':np.asarray(self.keys)}
        arrays.update({f'observation_{i}':np.stack([x[k] for x in self.observations]) for i,k in enumerate(self.keys)})
        if self.base_schema is not None:
            names,bounds,width=self.base_schema
            arrays['base_policy_term_names']=np.asarray(names)
            arrays['base_policy_term_slices']=np.asarray(bounds,dtype=np.int64)
            arrays['base_policy_dimension']=np.asarray(width)
        np.savez_compressed(path,**arrays)
