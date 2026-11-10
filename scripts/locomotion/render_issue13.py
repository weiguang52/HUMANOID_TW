"""Four precolored exact-state front/side replays of the two frozen median trials."""
import json,os
from pathlib import Path
from issue13_common import ROOT,REPO,run,sha
from run_issue13 import verify_frozen
def main():
 os.chdir(REPO);assert (ROOT/"reconstruction_complete").exists();verify_frozen()
 examples=json.loads((ROOT/"reconstruction_state.json").read_text());models=json.loads((ROOT/"input_models.json").read_text());plan=[]
 for group,s in examples.items():
  selection=ROOT/f"replay_selection_{group}.json";selection.write_text(json.dumps(s,indent=2))
  record={"group":group,"model":s["model"],"env_id":s["env_id"],"push_seconds":s["push_seconds"],"views":{}}
  m=models[s["model"]];assert sha(m["checkpoint"])==m["sha256"]
  for view in ["front","side"]:
   out=ROOT/f"{group}_{view}.json"
   if not out.exists():
    args=["python","-u","scripts/locomotion/render_eval_issue13.py","--task","Unitree-TW-30dof-Velocity","--headless",
     "--num_envs","1","--seed","1","--checkpoint",m["checkpoint"],"--experiment_profile","robust08",
     "--mass_randomization","bounded","--last_action_observation","requested","--issue5_condition","clean",
     "--output",str(out),"--video","--camera_view",view,"--trajectory_replay",str(selection)]
    run(args,out.with_suffix(".log"),"replay_"+group+"_"+view)
   d=json.loads(out.read_text());assert d["frames"]==1000 and d["fps"]==50 and not d["physics_integrated"]
   assert d["source_trajectory_sha256"]==s["trajectory_sha256"] and d["push_seconds"]==s["push_seconds"]
   assert "Colored USD integrity: 1876 original attributes unchanged" in out.with_suffix(".log").read_text()
   video=Path(d["video"]);assert video.exists()
   record["views"][view]={"path":str(video),"sha256":sha(video),"metadata":str(out),"source_trajectory_sha256":s["trajectory_sha256"]}
  plan.append(record);(ROOT/"video_plan.json").write_text(json.dumps(plan,indent=2))
 assert len(plan)==2
 (ROOT/"videos_complete").write_text("Four colored exact-state20s median-trial front/side replays with actual push timeline.\n")
 print("ISSUE13_VIDEOS_COMPLETE",flush=True)
if __name__=="__main__":main()
