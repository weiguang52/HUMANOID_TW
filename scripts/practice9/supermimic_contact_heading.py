"""Offline diagnostic heading features; no changes to policy/model loading."""
import torch
from supermimic_yaw_canonical import canonicalize_state
CONTRACT='world_contact_heading_fourfold_v1'
def contact_heading_features(state):
 """Retain contact heading modulo quarter turns, with continuous gradients.

 Native diagnostics support quarter-turn symmetry but disprove arbitrary yaw
 symmetry for current pyramidal contact. These features are a hypothesis for
 model calibration, not an accepted physical/control contract.
 """
 _,frame=canonicalize_state(state)
 c,s=frame['c'],frame['s'];c2=c*c-s*s;s2=2*c*s
 return torch.cat([c2*c2-s2*s2,2*c2*s2],dim=-1).to(state.dtype)
