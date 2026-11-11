"""Capture only the new median trial and reuse exact archived old representatives."""
import json,os
from pathlib import Path
import numpy as np
from issue14_common import ROOT,REPO,run,sha
from run_issue14 import verify_frozen,evaluate_args
from smoke_issue14 import compare_npz
OLD=Path("/root/gpufree-data/tw_stage1/issue13_20261110")
def main():
 os.chdir(REPO);verify_frozen();assert (ROOT/"analysis_complete").exists()
 selected=json.loads((ROOT/"representative_selection.json").read_text());assert sha(ROOT/"representative_selection.json")==json.loads((ROOT/"representative_selection_sha256.json").read_text())["sha256"]
 state=json.loads((OLD/"reconstruction_state.json").read_text())
 oldmanifest=json.loads((REPO/"validation_artifacts/stage1_issue13_20261110/sha256.json").read_text())
 for group,s in state.items():
  assert s["env_id"]==selected[group]["env_id"] and s["model"]==selected[group]["model"]
  for key in ["trajectory","trajectory_meta"]:
   p=Path(s[key]);assert sha(p)==oldmanifest[str(p.relative_to(OLD))]
  assert sha(s["trajectory"])==s["trajectory_sha256"] and s["all_original_metrics_raw_records_and_moments_exactly_reproduced"]
 group="clock";s=selected[group];model=json.loads((ROOT/"final_models.json").read_text())[s["model"]]
 out=ROOT/"reconstruction/clock/source.json";out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists():run(evaluate_args(model,s["eval_seed"],"combined",out,capture=s["env_id"]),out.with_suffix(".log"),"reconstruct_clock")
 d=json.loads(out.read_text());original=json.loads(Path(s["source_output"]).read_text())
 for key,v in original.items():
  if key!="issue13_tracking":assert d[key]==v,(key,"changed original output")
 for key,v in original["issue13_tracking"].items():
  if key!="output":assert d["issue13_tracking"][key]==v,key
 folder=Path(d["issue13_tracking"]["output"]);oldfolder=Path(original["issue13_tracking"]["output"])
 for row in d["results"]:
  cmd=row["command"];compare_npz(folder/(cmd+".npz"),oldfolder/(cmd+".npz"))
  for suffix in [".npz","_trials.npz"]:
   compare_npz(out.parent/"traces/source"/(cmd+suffix),Path(s["source_output"]).parent/"traces"/Path(s["source_output"]).stem/(cmd+suffix))
 trajectory=folder/"forward_08_trajectory.npz";meta=json.loads(trajectory.with_suffix(".json").read_text())
 with np.load(trajectory) as z,np.load(oldfolder/"forward_08.npz") as moments:
  assert z["values"].shape==(1000,43) and np.isfinite(z["values"]).all() and meta["env_id"]==s["env_id"]
  masks=np.unpackbits(moments["packed_masks"],axis=-1,count=1024,bitorder="little").astype(bool)
  assert np.array_equal(z["flags"][:,0],masks[:,0,0,s["env_id"]])
  assert np.array_equal(z["flags"][:,0]&z["flags"][:,2],masks[:,0,1,s["env_id"]])
  pushes=moments["push_steps_and_ids"];steps=pushes[pushes[:,1]==s["env_id"],0];assert np.array_equal(z["push_steps"],steps)
  reset=np.flatnonzero(~z["flags"][:,0]).tolist()
 state[group]={**s,"trajectory":str(trajectory),"trajectory_meta":str(trajectory.with_suffix(".json")),"trajectory_sha256":sha(trajectory),
  "push_seconds":(steps*.02).tolist(),"reset_frames":reset,"all_original_metrics_raw_records_and_moments_exactly_reproduced":True,"formal_denominators_unchanged":True}
 assert len(state)==3;verify_frozen();(ROOT/"reconstruction_state.json").write_text(json.dumps(state,indent=2))
 (ROOT/"reconstruction_complete").write_text("Old two representative trajectories reused unchanged; new median source reconstructed and exactly audited.\n")
 print("ISSUE14_RECONSTRUCTION_COMPLETE",flush=True)
if __name__=="__main__":main()
