"""Six colored representative clips: four exact old clips reused, two new exact-state renders."""
import json,os,shutil,subprocess
from pathlib import Path
from issue14_common import ROOT,REPO,BASE,run,sha
from run_issue14 import verify_frozen
OLD=Path("/root/gpufree-data/tw_stage1/issue13_20261110")
def main():
 os.chdir(REPO);verify_frozen();assert (ROOT/"reconstruction_complete").exists()
 plan=[];videos=ROOT/"videos";videos.mkdir(exist_ok=True);oldmanifest=json.loads((REPO/"validation_artifacts/stage1_issue13_20261110/sha256.json").read_text())
 for entry in json.loads((OLD/"video_plan.json").read_text()):
  record={"group":entry["group"],"model":entry["model"],"env_id":entry["env_id"],"push_seconds":entry["push_seconds"],"views":{},"reused_issue13":True}
  for view,e in entry["views"].items():
   src=Path(e["path"]);dest=videos/src.name;assert sha(src)==e["sha256"]==oldmanifest[str(src.relative_to(OLD))]
   shutil.copy2(src,dest);metadata=ROOT/Path(e["metadata"]).name;shutil.copy2(e["metadata"],metadata)
   archived_log=REPO/"validation_artifacts/stage1_issue13_20261110"/Path(e["metadata"]).with_suffix(".log").name
   assert sha(archived_log)==oldmanifest[archived_log.name];shutil.copy2(archived_log,ROOT/archived_log.name)
   assert sha(dest)==e["sha256"];record["views"][view]={**e,"path":str(dest),"metadata":str(metadata),"original_video":str(src),"bytes_reused_exactly":True}
  plan.append(record)
 reconstruction=json.loads((ROOT/"reconstruction_state.json").read_text());s=reconstruction["clock"];selection=ROOT/"replay_selection_clock.json";selection.write_text(json.dumps(s,indent=2))
 m=json.loads((ROOT/"final_models.json").read_text())[s["model"]];record={"group":"clock","model":s["model"],"env_id":s["env_id"],"push_seconds":s["push_seconds"],"views":{},"reused_issue13":False}
 for view in ["front","side"]:
  out=ROOT/f"clock_{view}.json"
  if not out.exists():
   args=["python","-u","scripts/locomotion/render_eval_issue13.py","--task","Unitree-TW-30dof-Velocity","--headless","--num_envs","1","--seed","1",
    "--checkpoint",m["checkpoint"],"--experiment_profile","robust08","--mass_randomization","bounded","--last_action_observation","requested",
    "--issue5_condition","clean","--output",str(out),"--video","--camera_view",view,"--trajectory_replay",str(selection)]
   run(args,out.with_suffix(".log"),"replay_clock_"+view)
  d=json.loads(out.read_text());assert d["frames"]==1000 and d["fps"]==50 and not d["physics_integrated"]
  assert d["source_trajectory_sha256"]==s["trajectory_sha256"] and d["push_seconds"]==s["push_seconds"]
  assert "Colored USD integrity: 1876 original attributes unchanged" in out.with_suffix(".log").read_text()
  p=Path(d["video"]);record["views"][view]={"path":str(p),"sha256":sha(p),"metadata":str(out),"source_trajectory_sha256":s["trajectory_sha256"]}
 plan.append(record);assert len(plan)==3;(ROOT/"video_plan.json").write_text(json.dumps(plan,indent=2))
 oldcolors=REPO/"validation_artifacts/stage1_issue13_20261110/joint_colors.json";assert sha(oldcolors)==oldmanifest["joint_colors.json"];shutil.copy2(oldcolors,ROOT/"joint_colors.json")
 shared=[Path("scripts/locomotion/replay_issue13.py"),Path("scripts/locomotion/render_eval_issue13.py"),Path("scripts/locomotion/eval_cmd.py")]
 for p in shared:assert p.read_bytes()==subprocess.check_output(["git","show",BASE+":"+str(p)])
 source=shared+[Path(__file__),Path("scripts/locomotion/reconstruct_issue14.py")]
 assets=[Path("/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3"+tail+".usd") for tail in ["","_colored"]]
 (ROOT/"media_source_sha256.json").write_text(json.dumps({"source_sha256":{str(p):sha(p) for p in source},"asset_sha256":{str(p):sha(p) for p in assets},"unchanged_shared_sources_match_base_commit":BASE,"video_plan_sha256":sha(ROOT/"video_plan.json"),"derived_code_not_claimed_prospectively_frozen":True},indent=2))
 (ROOT/"videos_complete").write_text("Six20s colored exact-state front/side videos, old four byte-identical, actual push timestamps.\n")
 print("ISSUE14_VIDEOS_COMPLETE",flush=True)
if __name__=="__main__":main()
