"""Reconstruct only the two prespecified median-yaw runs for exact source-state capture."""
import json,os
from pathlib import Path
import numpy as np
from issue15_common import ROOT,REPO,run,sha,evaluate_args
from run_issue15 import verify_frozen
from smoke_issue14 import compare_npz
def main():
 os.chdir(REPO);verify_frozen();assert (ROOT/"analysis_complete").exists()
 cohort=json.loads((ROOT/"cohort.json").read_text());selected=json.loads((ROOT/"representative_selection.json").read_text());assert len(selected)==2;state={}
 for s in selected:
  group=s["group"];model=cohort["unique_evaluation_models"][s["model"]];out=ROOT/"reconstruction"/group/"source.json";out.parent.mkdir(parents=True,exist_ok=True)
  if not out.exists():run(evaluate_args(model,s["eval_seed"],out,capture=s["env_id"]),out.with_suffix(".log"),"reconstruct_"+group)
  d=json.loads(out.read_text());original=json.loads(Path(s["output"]).read_text())
  for k,v in original.items():
   if k!="issue15_tracking":assert d[k]==v,(group,k)
  for k,v in original["issue15_tracking"].items():
   if k!="output":assert d["issue15_tracking"][k]==v,(group,k)
  folder=Path(d["issue15_tracking"]["output"]);oldfolder=Path(original["issue15_tracking"]["output"])
  for row in original["results"]:
   for suffix in [".npz","_yaw.npz"]:compare_npz(folder/(row["command"]+suffix),oldfolder/(row["command"]+suffix))
  trajectory=folder/"forward_03_trajectory.npz";meta=json.loads(trajectory.with_suffix(".json").read_text());assert meta["env_id"]==s["env_id"] and meta["command"]=="forward_03"
  with np.load(trajectory) as z,np.load(oldfolder/"forward_03.npz") as moments,np.load(oldfolder/"forward_03_yaw.npz") as yaw:
   assert z["values"].shape==(1000,43) and np.isfinite(z["values"]).all() and not len(z["push_steps"])
   masks=np.unpackbits(moments["packed_masks"],axis=-1,count=1024,bitorder="little").astype(bool)
   assert np.array_equal(z["flags"][:,0],masks[:,0,0,s["env_id"]])
   pose=z["values"];q=pose[:,3:7];angles=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
   assert np.allclose(angles,yaw["heading_yaw"][:,s["env_id"]],atol=1e-6)
   start=float(yaw["heading_start"][s["env_id"]]);resets=np.flatnonzero(~z["flags"][:,0]).tolist()
  state[group]={**s,"trajectory":str(trajectory),"trajectory_meta":str(trajectory.with_suffix(".json")),"trajectory_sha256":sha(trajectory),"heading_start_yaw":start,"push_seconds":[],"reset_frames":resets,"command":"forward_03","all_original_metrics_raw_records_and_moments_exactly_reproduced":True,"formal_denominators_unchanged":True}
 verify_frozen();(ROOT/"reconstruction_state.json").write_text(json.dumps(state,indent=2)+"\n");(ROOT/"reconstruction_complete").write_text("Two exact median-yaw trial reconstructions passed all original outputs/arrays, no new formal denominator.\n")
 print("ISSUE15_RECONSTRUCTION_COMPLETE",flush=True)
if __name__=="__main__":main()
