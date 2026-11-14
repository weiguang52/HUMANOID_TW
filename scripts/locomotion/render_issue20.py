"""Six exact-state colored views and synchronized paired-overhead delivery."""
import json,subprocess,os,time
from pathlib import Path
from issue20_common import ROOT,REPO,sha,save

def run(a,log,phase):
 with Path(log).open('w') as f:
  child=subprocess.Popen(a,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);save('render_progress.json',{'state':phase,'driver_pid':os.getpid(),'child_pid':child.pid,'started_at':time.time()});assert child.wait()==0,phase

def main():
 os.chdir(REPO);assert (ROOT/'formal_evaluation_complete').exists();selected=json.loads((ROOT/'representative_selection.json').read_text());assert sha(selected['checkpoint'])==selected['checkpoint_sha256']
 if not (ROOT/'colored_asset_integrity.json').exists():run(['python','-u','scripts/locomotion/color_issue20_usd.py'],ROOT/'color_usd.log','precolor_exact_selected_asset')
 plan=[]
 for phase in [0,10]:
  selection=ROOT/f'replay_selection_k{phase}.json';s=json.loads(selection.read_text());row={'phase':phase,'usd':s['usd'],'training_seed':s['training_seed'],'eval_seed':1,'env_id':0,'views':{}}
  for view in ['front','side','overhead']:
   output=ROOT/f'phase{phase}_{view}.json'
   if not output.exists():
    a=['python','-u','scripts/locomotion/render_eval_issue20.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1','--seed','1','--checkpoint',s['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--output',str(output),'--video','--camera_view',view,'--trajectory_replay',str(selection)];run(a,output.with_suffix('.log'),f'phase{phase}_{view}')
   d=json.loads(output.read_text());assert d['passed'] and d['phase']==phase and d['frames']==1000 and d['fps']==50 and not d['physics_integrated'];assert d['source_trajectory_sha256']==s['trajectory_sha256'];assert d['replay_source_sha256']==sha(REPO/'scripts/locomotion/replay_issue20.py');assert d['camera_follows_heading']==(view!='overhead');assert 'original attributes unchanged' in output.with_suffix('.log').read_text();assert sha(d['video'])==d['video_sha256'];row['views'][view]={'path':d['video'],'sha256':d['video_sha256'],'metadata':str(output)}
  plan.append(row)
 paired=ROOT/'videos/phase0_phase10_overhead_paired.mp4';a=plan[0]['views']['overhead']['path'];b=plan[1]['views']['overhead']['path']
 subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-y','-i',a,'-i',b,'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-r','50',str(paired)],check=True)
 save('video_plan.json',plan);save('paired_overhead_video.json',{'passed':True,'video':str(paired),'sha256':sha(paired),'left_phase':0,'right_phase':10,'source_videos':[a,b],'same_formal_frame_times':True,'physics_integrated':False,'world_XY_common_scale':True});save('media_source_sha256.json',{'source_files':{str(p):sha(p) for p in [REPO/'scripts/locomotion'/n for n in ['render_issue20.py','render_eval_issue20.py','replay_issue20.py','color_issue20_usd.py'] ]},'assets':json.loads((ROOT/'colored_asset_integrity.json').read_text()),'new_training_performed':False,'extra_evaluation_rollouts':False,'source_trajectory_SHA256_checked':True});(ROOT/'videos_complete').write_text('Both phases colored front/side/overhead plus paired overhead; exact formal states, no dynamics integration.\n');save('render_progress.json',{'state':'complete','driver_pid':os.getpid()});print('ISSUE20_VIDEOS_COMPLETE7',flush=True)
if __name__=='__main__':main()
