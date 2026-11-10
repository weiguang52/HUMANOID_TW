"""Capture the two frozen median trials, validating exact reproduction of formal data."""
import json,os
from pathlib import Path
import numpy as np
from issue13_common import ROOT,REPO,args_for,run,sha
from run_issue13 import verify_frozen,array_equal
def main():
 os.chdir(REPO);assert (ROOT/"analysis_complete").exists();verify_frozen()
 selection=json.loads((ROOT/"representative_selection.json").read_text())
 assert sha(ROOT/"representative_selection.json")==json.loads((ROOT/"representative_selection_sha256.json").read_text())["sha256"]
 statefile=ROOT/"reconstruction_state.json";state=json.loads(statefile.read_text()) if statefile.exists() else {}
 for group,s in selection.items():
  name=s["model"];es=s["eval_seed"];out=ROOT/"reconstruction"/group/"source.json";out.parent.mkdir(parents=True,exist_ok=True)
  if group not in state:
   assert not out.exists(),"Unaccounted reconstruction output; inspect before retrying"
   run(args_for(name,es,"combined",out,capture_env=s["env_id"]),out.with_suffix(".log"),"reconstruct_"+group)
  d=json.loads(out.read_text());original=json.loads(Path(s["source_output"]).read_text())
  for key,value in original.items():
   if key!="issue13_tracking":assert d[key]==value,(group,key)
  for key,value in original["issue13_tracking"].items():
   if key!="output":assert d["issue13_tracking"][key]==value,(group,key)
  folder=Path(d["issue13_tracking"]["output"]);old=Path(original["issue13_tracking"]["output"])
  for row in d["results"]:array_equal(folder/(row["command"]+".npz"),old/(row["command"]+".npz"))
  for row in d["results"]:
   command=row["command"]
   for suffix in [".npz","_trials.npz"]:
    array_equal(out.parent/"traces/source"/(command+suffix),Path(s["source_output"]).parent/"traces"/Path(s["source_output"]).stem/(command+suffix))
  trajectory=folder/"forward_08_trajectory.npz";meta=json.loads(trajectory.with_suffix(".json").read_text());z=np.load(trajectory)
  assert meta["env_id"]==s["env_id"] and meta["frames"]==1000 and meta["dt"]==.02
  assert np.isfinite(z["values"]).all() and z["values"].shape==(1000,43)
  moments=np.load(old/"forward_08.npz");masks=np.unpackbits(moments["packed_masks"],axis=-1,count=1024,bitorder="little").astype(bool)
  assert np.array_equal(z["flags"][:,0],masks[:,0,0,s["env_id"]])
  assert np.array_equal(z["flags"][:,0]&z["flags"][:,2],masks[:,0,1,s["env_id"]])
  push=moments["push_steps_and_ids"];steps=push[push[:,1]==s["env_id"],0]
  assert np.array_equal(z["push_steps"],steps)
  source={**s,"trajectory":str(trajectory),"trajectory_meta":str(trajectory.with_suffix(".json")),"trajectory_sha256":sha(trajectory),
   "push_seconds":(steps*.02).tolist(),"reset_frames":np.flatnonzero(~z["flags"][:,0]).tolist(),
   "all_original_metrics_raw_records_and_moments_exactly_reproduced":True,"formal_denominators_unchanged":True}
  state[group]=source;statefile.write_text(json.dumps(state,indent=2))
 assert len(state)==2;verify_frozen()
 (ROOT/"reconstruction_complete").write_text("Two median trial trajectories captured; all formal data exactly reproduced; no extra formal denominators or training.\n")
 print("ISSUE13_RECONSTRUCTION_COMPLETE",flush=True)
if __name__=="__main__":main()
