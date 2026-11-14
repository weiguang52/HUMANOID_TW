"""Wait for the required diagnostic queue, then analyze/audit/render once."""
import json,time,subprocess,os
from pathlib import Path
from issue21_common import ROOT,REPO,save,sha
files={str(REPO/'scripts/locomotion'/n):sha(REPO/'scripts/locomotion'/n) for n in ['issue21_analysis.py','analyze_issue21.py','classify_issue21.py','plot_issue21.py','report_issue21.py','audit_issue21.py','prepare_issue21_media.py','render_issue21.py','render_eval_issue21.py','replay_issue21.py','audit_issue21_videos.py']};save('frozen_analysis_presentation_sources.json',{'files':files,'frozen_before_any_results':False,'frozen_before_full_cohort':True,'completed_jobs_at_freeze':len(json.loads((ROOT/'evaluation_records.json').read_text())),'frozen_at':time.time()})
while not (ROOT/'formal_evaluation_complete').exists():
 p=json.loads((ROOT/'evaluation_progress.json').read_text());assert Path('/proc/%d'%p['driver_pid']).exists(),'Evaluation driver exited without completion; inspect failure before resume';save('postprocess_progress.json',{'state':'waiting_for_formal_diagnostics','driver_pid':os.getpid()});time.sleep(20)
for script in ['analyze_issue21.py','classify_issue21.py','audit_issue21.py','plot_issue21.py','prepare_issue21_media.py','render_issue21.py','audit_issue21_videos.py','report_issue21.py']:
 for p,d in files.items():assert sha(p)==d,p
 with (ROOT/(script.removesuffix('.py')+'.log')).open('w') as log:
  child=subprocess.Popen(['python','-u','scripts/locomotion/'+script],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('postprocess_progress.json',{'state':script,'driver_pid':os.getpid(),'child_pid':child.pid,'started_at':time.time()});assert child.wait()==0,script
save('postprocess_progress.json',{'state':'actual_manual_visual_review_required','driver_pid':os.getpid()});print('ISSUE21_READY_FOR_ACTUAL_REVIEW',flush=True)
