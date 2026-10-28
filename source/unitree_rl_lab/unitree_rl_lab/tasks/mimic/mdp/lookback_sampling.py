"""Prototype: split observed failure-score mass to same-motion preceding bin."""
import torch

def preceding_failure_score(score,visits,motion_ids,local_starts,share=.5):
 if not 0<=share<=1:raise ValueError('share outside[0,1]')
 if score.ndim!=1 or any(v.shape!=score.shape for v in [visits,motion_ids,local_starts]):raise ValueError('matching flat bin arrays required')
 if any(not torch.isfinite(v).all() for v in [score,visits,local_starts]) or (score<0).any() or (visits<0).any():raise ValueError('invalid scores/counts')
 n=len(score);indices=torch.arange(n,device=score.device);previous=(indices-1).clamp_min(0)
 eligible=(indices>0)&(motion_ids==motion_ids[previous])&(local_starts>local_starts[previous])&(visits>torch.finfo(visits.dtype).eps)
 moved=torch.where(eligible,score*share,0.);result=score-moved;result=result.scatter_add(0,previous,moved)
 return result

