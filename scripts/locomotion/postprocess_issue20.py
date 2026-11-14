"""Finish analysis presentation and media after the prescribed evaluation stop/completion."""
import os,json,time,subprocess
from pathlib import Path
from issue20_common import ROOT,REPO,sha,save
SCRIPTS=['report_issue20.py','plot_issue20.py','color_issue20_usd.py','render_issue20.py','render_eval_issue20.py','replay_issue20.py','audit_issue20_videos.py']
def main():
 os.chdir(REPO);sources={str(REPO/'scripts/locomotion'/n):sha(REPO/'scripts/locomotion'/n) for n in SCRIPTS};save('frozen_presentation_sources.json',{'files':sources,'frozen_before_aggregate_analysis':not (ROOT/'symmetric_analysis.json').exists(),'frozen_at':time.time()})
 while not (ROOT/'formal_evaluation_complete').exists():
  assert Path('/proc/120504').exists(),'Evaluation driver terminal before required completion; inspect actual failure before resuming'
  save('postprocess_progress.json',{'state':'waiting_for_formal_evaluation','driver_pid':os.getpid()});time.sleep(20)
 for script in ['report_issue20.py','plot_issue20.py','render_issue20.py','audit_issue20_videos.py']:
  for f,digest in sources.items():assert sha(f)==digest,f
  with (ROOT/(script.removesuffix('.py')+'.log')).open('w') as log:
   child=subprocess.Popen(['python','-u','scripts/locomotion/'+script],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('postprocess_progress.json',{'state':script,'driver_pid':os.getpid(),'child_pid':child.pid,'started_at':time.time()});assert child.wait()==0,script
 save('postprocess_progress.json',{'state':'actual_manual_plot_and_video_review_required','driver_pid':os.getpid(),'training_updates':0,'required_reviews':['4 scientific plots','2 contact sheets with42 single-view sampled frames','full-resolution arm/turn/final frames for both phases','paired overhead frames']});print('ISSUE20_POSTPROCESS_READY_FOR_ACTUAL_REVIEW',flush=True)
if __name__=='__main__':main()
