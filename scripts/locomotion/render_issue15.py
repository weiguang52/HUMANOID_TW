"""Four precolored median-yaw clips with exact-state replay and a world-heading inset."""
import json,os,shutil,subprocess
from pathlib import Path
from issue15_common import ROOT,REPO,BASE,run,sha
from run_issue15 import verify_frozen
def main():
 os.chdir(REPO);verify_frozen();assert (ROOT/"reconstruction_complete").exists()
 reconstruction=json.loads((ROOT/"reconstruction_state.json").read_text());plan=[]
 for group,s in reconstruction.items():
  selection=ROOT/("replay_selection_"+group+".json");selection.write_text(json.dumps(s,indent=2)+"\n")
  record={"group":group,"model":s["model"],"env_id":s["env_id"],"push_seconds":[],"views":{}}
  for view in ["front","side"]:
   out=ROOT/(group+"_"+view+".json")
   if not out.exists():
    args=["python","-u","scripts/locomotion/render_eval_issue15.py","--task","Unitree-TW-30dof-Velocity","--headless","--num_envs","1","--seed","1","--checkpoint",s["checkpoint"],"--experiment_profile","robust08","--mass_randomization","bounded","--last_action_observation","requested","--issue5_condition","clean","--output",str(out),"--video","--camera_view",view,"--trajectory_replay",str(selection)]
    run(args,out.with_suffix(".log"),"render_"+group+"_"+view)
   d=json.loads(out.read_text());assert d["frames"]==1000 and d["fps"]==50 and not d["physics_integrated"] and d["command"]=="forward_03"
   assert d["source_trajectory_sha256"]==s["trajectory_sha256"] and not d["push_seconds"]
   assert d["replay_source_sha256"]==sha("scripts/locomotion/replay_issue15.py") and d["render_adapter_sha256"]==sha("scripts/locomotion/render_eval_issue15.py")
   assert "Colored USD integrity: 1876 original attributes unchanged" in out.with_suffix(".log").read_text()
   p=Path(d["video"]);record["views"][view]={"path":str(p),"sha256":sha(p),"metadata":str(out),"source_trajectory_sha256":s["trajectory_sha256"]}
  plan.append(record)
 assert len(plan)==2;(ROOT/"video_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
 shutil.copy2(REPO/"validation_artifacts/stage1_issue14_20261111/joint_colors.json",ROOT/"joint_colors.json")
 shared=[Path("scripts/locomotion/eval_cmd.py"),Path("scripts/locomotion/replay_issue13.py"),Path("scripts/locomotion/render_eval_issue13.py")]
 for p in shared:assert p.read_bytes()==subprocess.check_output(["git","show",BASE+":"+str(p)])
 sources=shared+[Path(__file__),Path("scripts/locomotion/replay_issue15.py"),Path("scripts/locomotion/render_eval_issue15.py"),Path("scripts/locomotion/reconstruct_issue15.py")]
 assets=[Path("/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3"+s+".usd") for s in ["","_colored"]]
 prior_assets=json.loads((REPO/"validation_artifacts/stage1_issue14_20261111/media_source_sha256.json").read_text())["asset_sha256"]
 for p in assets:assert sha(p)==prior_assets[str(p)],"Experiment asset differs from prior committed source"
 (ROOT/"media_source_sha256.json").write_text(json.dumps({"sources":{str(p):sha(p) for p in sources},"assets":{str(p):sha(p) for p in assets},"video_plan_sha256":sha(ROOT/"video_plan.json"),"pose_replay_no_physics_integration":True,"display_joint_velocity_zero":True,"camera_follows_translation_and_heading":True,"world_XY_trail_and_heading_inset":True,"humanml_source_npy":"N/A: Stage1 velocity commands"},indent=2)+"\n")
 (ROOT/"videos_complete").write_text("Four20s exact-state colored front/side clips with explicit world heading/trail inset.\n")
 print("ISSUE15_VIDEOS_COMPLETE",flush=True)
if __name__=="__main__":main()
