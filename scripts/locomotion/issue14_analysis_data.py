"""Independent reconstruction of issue13 packed windows and sufficient statistics."""
import csv,gzip,hashlib,io,json
from pathlib import Path
import numpy as np
from issue14_common import ROOT
from issue13_tracking import FIELDS,SCOPES,PARTS
def load_command(folder,command,result):
 folder=Path(folder);meta=json.loads((folder/(command+".json")).read_text())
 z=dict(np.load(folder/(command+".npz")));n=meta["num_envs"];T=meta["steps"]
 assert n==1024 and T==1000 and meta["dt"]==.02
 c,s,q=z["counts"],z["sums"],z["sumsq"]
 assert c.shape==(2,5,n) and s.shape==q.shape==(2,5,n,5)
 assert np.isfinite(s).all() and np.isfinite(q).all() and (c>=0).all() and (c<=T).all()
 masks=np.unpackbits(z["packed_masks"],axis=-1,count=n,bitorder="little").astype(bool)
 assert masks.shape==(T,2,5,n)
 assert np.array_equal(masks.sum(0),c) and np.array_equal(masks.sum(-1),z["step_counts"])
 valid=masks[:,0,0];alive=np.ones(n,dtype=bool);last=np.full(n,-2**60,dtype=np.int64)
 pushes=z["push_steps_and_ids"];assert pushes.ndim==2 and pushes.shape[1]==2
 if len(pushes):
  assert ((pushes[:,0]>=1)&(pushes[:,0]<=T)&(pushes[:,1]>=0)&(pushes[:,1]<n)).all()
  assert len(np.unique(pushes,axis=0))==len(pushes)
 bystep={k:pushes[pushes[:,0]==k,1] for k in np.unique(pushes[:,0])}
 for t in range(T):
  step=t+1;last[~valid[t]]=-2**60
  if step in bystep:last[bystep[step]]=step
  age=step-last;inside=(age>=0)&(age<50);ri=(age>0)&(age<=50)
  partitions=np.stack([np.ones(n,dtype=bool),inside,~inside,ri,~ri])
  alive &=valid[t]
  expected=np.stack([partitions&valid[t],partitions&valid[t]&alive])
  assert np.array_equal(expected,masks[t]),(command,step,"window/reset reconstruction")
 assert np.array_equal(c[:,0],c[:,1]+c[:,2]) and np.array_equal(c[:,0],c[:,3]+c[:,4])
 for a in [s,q]:
  for i,j in [(1,2),(3,4)]:assert np.allclose(a[:,0],a[:,i]+a[:,j],atol=1e-9,rtol=1e-11)
 assert np.allclose(s.sum(2),z["step_sums"].sum(0),atol=1e-8,rtol=1e-10)
 assert np.allclose(q.sum(2),z["step_sumsq"].sum(0),atol=1e-8,rtol=1e-10)
 refs=z["reference_env_ids"];values=z["reference_values"].astype("float64");rm=z["reference_masks"]
 assert np.array_equal(rm,masks[:,:,:,refs])
 for si in range(2):
  for pi in range(5):
   for ri,env in enumerate(refs):
    a=values[:,ri][rm[:,si,pi,ri]]
    assert len(a)==c[si,pi,env]
    assert np.allclose(a.sum(0),s[si,pi,env],atol=1e-9,rtol=1e-11)
    assert np.allclose(np.square(a).sum(0),q[si,pi,env],atol=1e-9,rtol=1e-11)
 count=c[0,0].sum();recomputed=s[0,0].sum(0)/count
 for j,key in enumerate(FIELDS[:2]):assert abs(recomputed[j]-result["metrics"][key])<1e-10
 assert int((z["first_failure_step"]>=0).sum())==result["failed_trials"]
 return z,meta
def env_statistics(z,si,pi,target):
 c=z["counts"][si,pi];s=z["sums"][si,pi];q=z["sumsq"][si,pi]
 means=np.divide(s,c[:,None],out=np.full_like(s,np.nan),where=c[:,None]>0)
 variance=np.divide(q,c[:,None],out=np.full_like(q,np.nan),where=c[:,None]>0)-means**2
 assert not np.any(variance[c>0]<-1e-10)
 std=np.sqrt(np.maximum(variance,0))
 return c,means,std
def pooled_statistics(z,si,pi,target):
 c,m,sd=env_statistics(z,si,pi,target);s=z["sums"][si,pi];q=z["sumsq"][si,pi]
 total=int(c.sum());all_total=int(z["counts"][si,0].sum())
 assert total>0 and all_total>0
 mean=s.sum(0)/total;pooled_var=np.maximum(q.sum(0)/total-mean**2,0)
 within=np.nansum(c*sd[:,2]**2)/total
 between=np.nansum(c*(m[:,2]-mean[2])**2)/total
 assert abs(within+between-pooled_var[2])<1e-9
 return {**{key:float(mean[i]) for i,key in enumerate(FIELDS[:2])},
  "xy_equal_trial_mean":float(np.nanmean(m[:,0])),"yaw_equal_trial_mean":float(np.nanmean(m[:,1])),
  "vx_mean":float(mean[2]),"vy_mean":float(mean[3]),"wz_mean":float(mean[4]),
  "vx_bias":float(mean[2]-target[0]),
  "vx_temporal_sd_mean":float(np.nanmean(sd[:,2])),
  "vy_temporal_sd_mean":float(np.nanmean(sd[:,3])),
  "wz_temporal_sd_mean":float(np.nanmean(sd[:,4])),
  "vx_pooled_sd":float(np.sqrt(pooled_var[2])),
  "vx_within_variance":float(within),"vx_between_variance":float(between),
  "valid_samples":total,"environments_with_samples":int((c>0).sum()),
  "sample_fraction":total/all_total,
  "xy_weighted_contribution":float(s[:,0].sum()/all_total),
  "yaw_weighted_contribution":float(s[:,1].sum()/all_total)}
class TrialCsvs:
 def __init__(self):
  self.streams={};self.writers={};self.rows={};self.paths={}
 def write(self,condition,command,base,z,target):
  key=condition+"_"+command
  if key not in self.writers:
   p=ROOT/"per_trial"/(key+".csv.gz");p.parent.mkdir(exist_ok=True)
   raw=p.open("wb");gz=gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0)
   text=io.TextIOWrapper(gz,newline="")
   self.streams[key]=(text,gz,raw);self.paths[key]=p;self.rows[key]=0
   fields=["condition","command","model","group","training_seed","eval_seed","env_id","scope","partition","samples","xy_error","yaw_error","vx_mean","vx_bias","vx_temporal_sd","vy_mean","vy_temporal_sd","wz_mean","wz_temporal_sd","first_failure_step"]
   w=csv.DictWriter(text,fieldnames=fields,lineterminator="\n");w.writeheader();self.writers[key]=w
  for pi in ([0] if condition=="clean" else [0,1,2]):
   c,m,sd=env_statistics(z,0,pi,target)
   for env in range(len(c)):
    valid=c[env]>0
    def v(x):return float(x) if valid else None
    self.writers[key].writerow({"condition":condition,"command":command,**base,"env_id":env,"scope":SCOPES[0],"partition":PARTS[pi],"samples":int(c[env]),
     "xy_error":v(m[env,0]),"yaw_error":v(m[env,1]),"vx_mean":v(m[env,2]),"vx_bias":v(m[env,2]-target[0]),
     "vx_temporal_sd":v(sd[env,2]),"vy_mean":v(m[env,3]),"vy_temporal_sd":v(sd[env,3]),
     "wz_mean":v(m[env,4]),"wz_temporal_sd":v(sd[env,4]),"first_failure_step":int(z["first_failure_step"][env])})
    self.rows[key]+=1
 def close(self):
  out={}
  for k,(text,gz,raw) in self.streams.items():
   text.close();raw.close();p=self.paths[k]
   out[str(p.relative_to(ROOT))]={"rows":self.rows[k],"gzip_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
    "decompressed_sha256":hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest(),
    "scope":"Human-readable primary per-trial rows; both scopes and both boundary conventions retained losslessly in raw moment NPZs."}
  return out
