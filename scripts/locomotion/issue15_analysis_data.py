"""Independent issue15 reconstruction of every forward yaw/heading record."""
import json
from pathlib import Path
import numpy as np
from issue13_analysis_data import load_command,pooled_statistics
from issue15_math import bias_variance
def load_yaw(folder,command,result):
 folder=Path(folder);z,meta=load_command(folder,command,result)
 y=dict(np.load(folder/(command+"_yaw.npz")));n=meta["num_envs"];T=meta["steps"]
 masks=np.unpackbits(z["packed_masks"],axis=-1,count=n,bitorder="little").astype(bool)
 assert np.array_equal(y["heading_count"],z["counts"][:,0])
 assert np.array_equal(y["first_failure_step"],z["first_failure_step"])
 refs=y["reference_env_ids"];record=y["heading_reference_yaw_delta"]
 assert record.shape==(T,len(refs),2)
 # Independent reference heading-delta accumulation for both scopes.
 for si in range(2):
  rec=np.where(masks[:,si,0][:,refs],record[:,:,1],0.).sum(0,dtype=np.float64)
  assert np.allclose(rec,y["heading_delta_sum"][si,refs],atol=2e-5,rtol=1e-6)
 if command.startswith("forward"):
  wz=y["wz"].astype(np.float64);heading=y["heading_yaw"].astype(np.float64)
  assert wz.shape==heading.shape==(T,n) and np.isfinite(wz).all() and np.isfinite(heading).all()
  prev=np.concatenate([y["heading_start"][None].astype(np.float64),heading[:-1]],axis=0)
  delta=np.arctan2(np.sin(heading-prev),np.cos(heading-prev))
  for si in range(2):
   mask=masks[:,si,0]
   expected=np.where(mask,wz,0.).sum(0);squares=np.where(mask,wz*wz,0.).sum(0)
   assert np.allclose(expected,z["sums"][si,0,:,4],atol=1e-8,rtol=1e-10)
   assert np.allclose(squares,z["sumsq"][si,0,:,4],atol=1e-8,rtol=1e-10)
   assert np.allclose(np.where(mask,abs(wz),0.).sum(0),z["sums"][si,0,:,1],atol=1e-8,rtol=1e-10)
   assert np.allclose(np.where(mask,delta,0.).sum(0),y["heading_delta_sum"][si],atol=1e-4,rtol=1e-6)
  contacts=np.unpackbits(y["packed_foot_contacts"],axis=1,count=n,bitorder="little").astype(bool)
  assert contacts.shape==(T,n,2)
 return z,y,masks,meta
def job_statistics(z,y,meta,result):
 target=meta["target"];out={}
 for si,scope in enumerate(["full","initial"]):
  p=pooled_statistics(z,si,0,target);b=bias_variance(z["counts"][si,0],z["sums"][si,0,:,4],z["sumsq"][si,0,:,4],target[2])
  for k,v in p.items():out[scope+"_"+k]=v
  for k,v in b.items():out[scope+"_yaw_"+k]=v
 out["fall_rate"]=result["failed_trials"]/result["num_envs"]
 survived=y["first_failure_step"]<0;degrees=y["heading_delta_sum"]*180/np.pi
 out["full20s_heading_survivors"]=int(survived.sum());out["original_trial_count"]=meta["num_envs"]
 for kind,values in [("full20s_survivor",degrees[0,survived]),("initial_prefix_all",degrees[1]),("all_episode_segments",degrees[0])]:
  for name,value in [("signed_mean",np.mean(values) if len(values) else None),("mean_abs",np.mean(abs(values)) if len(values) else None),("median_abs",np.median(abs(values)) if len(values) else None),("p95_abs",np.percentile(abs(values),95) if len(values) else None)]:
   out[kind+"_heading_"+name+"_degrees"]=None if value is None else float(value)
 out["initial_prefix_mean_duration_s"]=float(y["heading_count"][1].mean()*meta["dt"])
 return out
