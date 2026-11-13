import json,os,subprocess,time
from pathlib import Path
root=Path('/root/gpufree-data/tw_stage1/issue17_20261113');c=json.loads(Path('validation_artifacts/stage1_issue16_20261111/cohort.json').read_text());parent=Path(c['models']['requested4500_s1']['checkpoint'])
for mode in ['original','disabled']:
 folder=root/'preflight'/mode;folder.mkdir(parents=True,exist_ok=True)
 args=['python','-u','scripts/locomotion/preflight_train_issue17.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','4096','--max_iterations','2','--run_name','issue17_disabled_reproduction','--seed','1','--resume','--load_run',parent.parent.name,'--checkpoint',parent.name,'--experiment_profile','robust08','--issue5_training_mode','strengthen','--issue5_factor','push','--mass_randomization','bounded','--last_action_observation','requested','--post_push_pitch_weight','0.0','--issue12_audit_dir',str(folder/'audit')]
 env=dict(os.environ,TW_ISSUE17_PREFLIGHT_MODE=mode)
 with (folder/'console.log').open('w') as out:
  child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17',*args],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  (root/'preflight_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),mode=mode,child_pid=child.pid,log=str(folder/'console.log'))))
  code=child.wait()
 assert code==0,(mode,code)
 print('PREFLIGHT_TWO_UPDATES_COMPLETE',mode,flush=True)
(root/'preflight_runs_complete').write_text('Both actual4096env2update runs ended successfully; bitwise comparison remains.\n')
