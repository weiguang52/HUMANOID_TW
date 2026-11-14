"""Three precolored views of the trained sym_wrap0.5 median-heading trial."""
import json,os,shutil,subprocess
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
BASE='c073475561787be7b81a87f4fa7af7fc46fd86de'
def sha(p):return __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()
def verify_frozen():
 for name in ['frozen_training_sources.json','frozen_evaluation_sources.json']:
  for p,h in json.loads((ROOT/name).read_text())['files'].items():assert sha(p)==h,p
 for row in json.loads((ROOT/'training_records.json').read_text()):assert sha(row['checkpoint'])==row['sha256']

def run(args,log,phase):
 with Path(log).open('w') as out:
  child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17render',*args],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  (ROOT/'render_progress.json').write_text(json.dumps(dict(phase=phase,child_pid=child.pid,log=str(log))))
  assert child.wait()==0,phase

def main():
 os.chdir(REPO);verify_frozen()
 s=json.loads((ROOT/'reconstruction_state.json').read_text())
 assert s['all_prior_original_recorded_arrays_bitwise_equal'] and s['all_prior_original_result_rows_exactly_reproduced']
 selection=ROOT/'replay_selection_sym_wrap.json';selection.write_text(json.dumps(s,indent=2)+'\n')
 record=dict(group='sym_wrap',model=s['model'],env_id=s['env_id'],push_seconds=[],selection_metric=s['selection_metric'],source_policy_mode='trained_symmetric_actor',views={})
 assets=[Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3'+suffix+'.usd') for suffix in ['','_colored']]
 prior=json.loads((REPO/'validation_artifacts/stage1_issue14_20261111/media_source_sha256.json').read_text())['asset_sha256']
 for p in assets:assert sha(p)==prior[str(p)],str(p)
 for view in ['front','side','overhead']:
  output=ROOT/('sym_wrap_'+view+'.json')
  if not output.exists():
   args=['python','-u','scripts/locomotion/render_eval_issue19.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1','--seed','1','--checkpoint',s['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--output',str(output),'--video','--camera_view',view,'--trajectory_replay',str(selection)]
   run(args,output.with_suffix('.log'),'render_sym_wrap_'+view)
  d=json.loads(output.read_text())
  assert d['frames']==1000 and d['fps']==50 and d['seconds']==20 and d['command']=='forward_05' and not d['physics_integrated']
  assert d['source_trajectory_sha256']==s['trajectory_sha256'] and not d['push_seconds']
  assert d['replay_source_sha256']==sha(REPO/'scripts/locomotion/replay_issue19.py')
  assert d['render_adapter_sha256']==sha(REPO/'scripts/locomotion/render_eval_issue19.py')
  assert d['camera_follows_heading']==(view!='overhead') and d['camera_follows_translation']
  assert 'Colored USD integrity: 1876 original attributes unchanged' in output.with_suffix('.log').read_text()
  p=Path(d['video']);record['views'][view]=dict(path=str(p),sha256=sha(p),metadata=str(output),source_trajectory_sha256=s['trajectory_sha256'])
 (ROOT/'video_plan.json').write_text(json.dumps([record],indent=2)+'\n')
 shutil.copy2(REPO/'validation_artifacts/stage1_issue14_20261111/joint_colors.json',ROOT/'joint_colors.json')
 sources=[REPO/'scripts/locomotion'/name for name in ['eval_cmd.py','render_eval_issue19.py','replay_issue19.py','reconstruct_issue19.py','render_issue19.py']]
 assert sources[0].read_bytes()==subprocess.check_output(['git','show',BASE+':scripts/locomotion/eval_cmd.py'])
 binding=dict(sources={str(p):sha(p) for p in sources},assets={str(p):sha(p) for p in assets},video_plan_sha256=sha(ROOT/'video_plan.json'),
  pose_replay_no_physics_integration=True,display_joint_velocity_zero=True,front_follows_translation_and_heading=True,overhead_follows_translation_fixed_world_heading=True,
  strict_projected_policy_source=True,new_training_performed=True,source_integer_clock=True,world_XY_trail_and_heading_inset=True,humanml_source_npy='N/A: Stage1 velocity commands')
 (ROOT/'media_source_sha256.json').write_text(json.dumps(binding,indent=2)+'\n')
 (ROOT/'videos_complete').write_text('Three20s precolored front/side/near-overhead views of the exact strict sym_wrap forward0.5 median-absolute-heading trial.\n')
 print('ISSUE17_VIDEOS_COMPLETE3VIEWS',flush=True)
if __name__=='__main__':main()
