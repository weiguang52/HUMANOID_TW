from pathlib import Path
import subprocess,time,sys,traceback,json
b=Path(__file__).parent
try:
 (b/'final_audit_status').write_text('waiting_numeric_queue')
 while True:
  q=b/'queue_status';status=q.read_text().strip() if q.exists() else ''
  live=subprocess.run(['tmux','has-session','-t','velocity_balance_queue'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
  if status=='failed_requires_inspection':raise RuntimeError('numeric queue failed')
  if status=='numeric_outputs_complete_pending_full_audit' and not live:break
  if not live:raise RuntimeError('numeric queue handle missing '+status)
  time.sleep(15)
 (b/'final_audit_status').write_text('auditing_full65')
 subprocess.run([sys.executable,str(b/'audit_evaluation.py')],check=True)
 report=json.load(open(b/'two_training_seed_report.json'));assert report['complete']
 (b/'final_audit_status').write_text('numeric_pass_visual_pending' if report['both_training_seeds_numeric_pass'] else 'numeric_rejected_no_promotion')
except Exception:
 traceback.print_exc();(b/'final_audit_status').write_text('failed_requires_inspection');raise
