"""Read-only fit windows for retaining measured dynamics during online updates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

class AnchorWindows:
    def __init__(self, root, window, state_names, action_names, device, seed):
        if window < 1:
            raise ValueError('Anchor window must be positive')
        root = Path(root).resolve()
        mapping_path = root / 'control_mapping.json'
        mapping = json.loads(mapping_path.read_text())
        if mapping['state_joint_names'] != list(state_names) or mapping['action_joint_names'] != list(action_names):
            raise ValueError('Anchor joint/action ordering mismatch')
        rows = json.loads((root / 'results.json').read_text())['rows']
        paths = [(root / row['trace']).resolve() for row in rows]
        if any(not path.is_relative_to(root) for path in paths):
            raise ValueError('Anchor trace escapes collection')
        fit_paths = [path for row,path in zip(rows,paths) if row['split']=='fit']
        other_paths = {path for row,path in zip(rows,paths) if row['split']!='fit'}
        if not fit_paths or len(set(fit_paths)) != len(fit_paths) or set(fit_paths) & other_paths:
            raise ValueError('Missing, repeated or split-overlapping fit traces')
        states, actions, contexts, provenance, times, initials = [], [], [], [], [], []
        for path in fit_paths:
            with np.load(path, allow_pickle=False) as z:
                s,n,u,c = [z[key].copy() for key in ['state','next_state','action','context']]
            count = len(s)
            expected = [(count,73),(count,73),(count,30),(count,34)]
            if any(x.shape != shape or not np.isfinite(x).all() for x,shape in zip([s,n,u,c],expected)):
                raise ValueError('Invalid anchor dimensions or nonfinite values')
            if not np.array_equal(n[:-1],s[1:]):
                raise ValueError('Anchor crosses reset or has noncontiguous states')
            if not np.array_equal(c,np.broadcast_to(c[0],c.shape)):
                raise ValueError('Anchor context changes inside episode')
            initials.append(s[0])
            # Same k+1 stored-state convention as online replay. Never pad or
            # include a terminal next_state that is not a stored live state.
            for start in range(count-window):
                states.append(s[start:start+window+1])
                actions.append(u[start:start+window+1])
                contexts.append(c[start+window])
                times.append(start+window)
            provenance.append(dict(file=path.name,frames=count,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if not states:
            raise ValueError('No complete anchor windows')
        self.states, self.actions, self.contexts = [torch.tensor(np.stack(x),device=device,dtype=torch.float32) for x in [states,actions,contexts]]
        self.end_times = torch.tensor(times,device=device,dtype=torch.long)
        self.initial_states = torch.tensor(np.stack(initials),device=device,dtype=torch.float32)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.metadata = dict(contract='fit_contiguous_fixed_context_v1',collection=str(root),window=window,windows=len(states),fit_files=provenance,mapping_sha256=hashlib.sha256(mapping_path.read_bytes()).hexdigest(),scope='Fit-only synthetic development windows; no fresh source-generalization claim')
        self.mapping = mapping

    def sample(self, count):
        ids = torch.randint(len(self.states),(count,),device=self.states.device,generator=self.generator)
        return self.states[ids], self.actions[ids], self.contexts[ids]

    def sample_with_times(self, count):
        ids = torch.randint(len(self.states),(count,),device=self.states.device,generator=self.generator)
        return self.states[ids], self.actions[ids], self.contexts[ids], self.end_times[ids]


def policy_history_parts(states, actions, history_len):
    """h+2 states include one extra past action, h past observations and current."""
    if states.shape[1] != history_len+2 or actions.shape[1] != history_len+2:
        raise ValueError('Policy anchor requires exactly h+2 contiguous stored frames')
    return states[:, -1], actions[:, -2], states[:, 1:-1], actions[:, :-2]
