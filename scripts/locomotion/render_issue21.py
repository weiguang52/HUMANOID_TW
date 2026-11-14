"""Two commands, both phases, front and side; saved states only."""
import json,subprocess,time,os
from issue21_common import ROOT,REPO,sha,save

def main():
 assert (ROOT/'formal_evaluation_complete').exists();selections=json.loads((ROOT/'media_selection.json').read_text());sources={str(REPO/'scripts/locomotion'/n):sha(REPO/'scripts/locomotion'/n) for n in ['render_issue21.py','render_eval_issue21.py','replay_issue21.py']};save('frozen_media_sources.json',sources);plan=[]
 for s in selections:
  selection=ROOT/('replay_%s_k%d.json'%(s['command'],s['phase']));row={'command':s['command'],'phase':s['phase'],'views':{}}
  for view in ['front','side']:
   for p,d in sources.items():assert sha(p)==d
   out=ROOT/('video_%s_k%d_%s.json'%(s['command'],s['phase'],view))
   if not out.exists():
    a=['python','-u','scripts/locomotion/render_eval_issue21.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1','--seed','1','--checkpoint',s['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--output',str(out),'--video','--camera_view',view,'--trajectory_replay',str(selection)]
    with out.with_suffix('.log').open('w') as log:
     child=subprocess.Popen(a,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('render_progress.json',{'driver_pid':os.getpid(),'child_pid':child.pid,'job':out.stem,'started_at':time.time()});assert child.wait()==0
   m=json.loads(out.read_text());assert m['passed'] and m['frames']==1000 and not m['physics_integrated'] and m['source_trajectory_sha256']==s['trajectory_sha256'];assert sha(m['video'])==m['video_sha256'];row['views'][view]={'path':m['video'],'sha256':m['video_sha256'],'metadata':str(out)}
  plan.append(row);save('video_plan.json',plan)
 (ROOT/'videos_complete').write_text('8 colored front/side exact-state clips\n');save('render_progress.json',{'state':'complete','driver_pid':os.getpid()});print('ISSUE21_VIDEOS_COMPLETE8',flush=True)
if __name__=='__main__':main()
