from pathlib import Path
import subprocess,time,json,sys,traceback
b=Path(__file__).parent;states=[b,b.parent/'anchor_strength_seed123_20261028']
try:
 (b/'video_queue_status').write_text('waiting_full_numeric_audit')
 while True:
  p=b/'final_audit_status';status=p.read_text().strip() if p.exists() else ''
  if status in ('numeric_pass_visual_pending','numeric_rejected_no_promotion'):break
  if status=='failed_requires_inspection':raise RuntimeError('full numeric audit failed')
  if subprocess.run(['tmux','has-session','-t','anchor_strength_final_audit'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:raise RuntimeError('full numeric audit handle missing')
  time.sleep(15)
 x=json.load(open(b/'two_training_seed_report.json'));assert x['complete']
 if not x['both_training_seeds_numeric_pass']:
  for s in states:(s/'videos/status').write_text('numeric_rejected_no_render')
  (b/'video_queue_status').write_text('numeric_rejected_no_render');sys.exit(0)
 for seed,gpu in [(42,0),(123,1)]:subprocess.run(['tmux','new-session','-d','-s','anchor_strength_videos'+str(seed),'bash '+str(b/'render_seed.sh')+' '+str(seed)+' '+str(gpu)],check=True)
 (b/'video_queue_status').write_text('renderers_started_visual_review_required')
 for seed,state in zip((42,123),states):
  while True:
   status=(state/'videos/status').read_text().strip()
   live=subprocess.run(['tmux','has-session','-t','anchor_strength_videos'+str(seed)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
   if status=='failed':raise RuntimeError('renderer failed '+state.name)
   if status=='rendered_pending_startup_review' and not live:break
   if not live:raise RuntimeError('renderer missing '+state.name+' '+status)
   time.sleep(10)
 subprocess.run([sys.executable,str(b/'audit_startup_frames.py')],check=True)
 startup=json.load(open(b/'startup_frame_audit.json'));assert startup['complete'] and startup['observed']==92 and not startup['blank_cases']
 for state in states:
  subprocess.run([sys.executable,str(b/'build_video_index.py'),str(state)],check=True)
  (state/'videos/status').write_text('rendered_pending_visual_review')
 (b/'video_queue_status').write_text('all92_technical_complete_visual_pending')
except Exception:
 traceback.print_exc();(b/'video_queue_status').write_text('failed_requires_inspection');raise
