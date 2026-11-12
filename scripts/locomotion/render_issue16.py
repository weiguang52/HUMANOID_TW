"""Three precolored views of the original control0.5 median-heading trial."""
import json,os,shutil,subprocess
from pathlib import Path
from issue16_common import ROOT,REPO,BASE,run,sha
from run_issue16 import verify_frozen

def main():
 os.chdir(REPO);verify_frozen()
 s=json.loads((ROOT/'reconstruction_state.json').read_text())
 assert s['all_prior_original_recorded_arrays_bitwise_equal'] and s['all_prior_original_result_rows_exactly_reproduced']
 selection=ROOT/'replay_selection_control6000.json';selection.write_text(json.dumps(s,indent=2)+'\n')
 record=dict(group='control6000',model=s['model'],env_id=s['env_id'],push_seconds=[],selection_metric=s['selection_metric'],source_policy_mode='original',views={})
 assets=[Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3'+suffix+'.usd') for suffix in ['','_colored']]
 prior=json.loads((REPO/'validation_artifacts/stage1_issue14_20261111/media_source_sha256.json').read_text())['asset_sha256']
 for p in assets:assert sha(p)==prior[str(p)],str(p)
 for view in ['front','side','overhead']:
  output=ROOT/('control6000_'+view+'.json')
  if not output.exists():
   args=['python','-u','scripts/locomotion/render_eval_issue16.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1','--seed','1','--checkpoint',s['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--output',str(output),'--video','--camera_view',view,'--trajectory_replay',str(selection)]
   run(args,output.with_suffix('.log'),'render_control6000_'+view)
  d=json.loads(output.read_text())
  assert d['frames']==1000 and d['fps']==50 and d['seconds']==20 and d['command']=='forward_05' and not d['physics_integrated']
  assert d['source_trajectory_sha256']==s['trajectory_sha256'] and not d['push_seconds']
  assert d['replay_source_sha256']==sha(REPO/'scripts/locomotion/replay_issue16.py')
  assert d['render_adapter_sha256']==sha(REPO/'scripts/locomotion/render_eval_issue16.py')
  assert d['camera_follows_heading']==(view!='overhead') and d['camera_follows_translation']
  assert 'Colored USD integrity: 1876 original attributes unchanged' in output.with_suffix('.log').read_text()
  p=Path(d['video']);record['views'][view]=dict(path=str(p),sha256=sha(p),metadata=str(output),source_trajectory_sha256=s['trajectory_sha256'])
 (ROOT/'video_plan.json').write_text(json.dumps([record],indent=2)+'\n')
 shutil.copy2(REPO/'validation_artifacts/stage1_issue14_20261111/joint_colors.json',ROOT/'joint_colors.json')
 sources=[REPO/'scripts/locomotion'/name for name in ['eval_cmd.py','render_eval_issue16.py','replay_issue16.py','reconstruct_issue16.py','render_issue16.py']]
 assert sources[0].read_bytes()==subprocess.check_output(['git','show',BASE+':scripts/locomotion/eval_cmd.py'])
 binding=dict(sources={str(p):sha(p) for p in sources},assets={str(p):sha(p) for p in assets},video_plan_sha256=sha(ROOT/'video_plan.json'),
  pose_replay_no_physics_integration=True,display_joint_velocity_zero=True,front_side_follow_translation_and_heading=True,overhead_follows_translation_fixed_world_heading=True,
  original_policy_source=True,world_XY_trail_and_heading_inset=True,humanml_source_npy='N/A: Stage1 velocity commands')
 (ROOT/'media_source_sha256.json').write_text(json.dumps(binding,indent=2)+'\n')
 (ROOT/'videos_complete').write_text('Three20s precolored front/side/near-overhead views of the exact original control0.5 median-absolute-heading trial.\n')
 print('ISSUE16_VIDEOS_COMPLETE3VIEWS',flush=True)
if __name__=='__main__':main()
