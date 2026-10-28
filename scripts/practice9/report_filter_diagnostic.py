"""Summarize fixed filter diagnostics without promoting partial or terminated trials."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);report={};limits={'static':109.1526597738266,'upper':35.05556285381317}
for case,limit in limits.items():
 p=root/'original'/f'{case}.json'
 if not p.exists():continue
 j=json.loads(p.read_text());d=np.load(p.with_suffix('.npz'));complete=j['steps']==1000 and not any(j['termination_counts'].values())
 video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate','-of','json',str(p.with_suffix('.mp4'))]))['streams'][0]
 drift=float(np.linalg.norm(d['anchor_actual'][-1,:2]-d['anchor_actual'][50,:2])*1000) if len(d['anchor_actual'])>50 else None
 report[case]={'seconds':j['seconds'],'complete_20s':complete,'terminations':j['termination_counts'],'drift_mm':drift,'drift_comparable':complete,'original_drift_mm':limit,'improves_original':complete and drift<limit,'video':video,'filter':j.get('action_filter')}
report['all_diagnostics_complete']=all(c in report for c in limits)
report['diagnostic_gate']=report['all_diagnostics_complete'] and all(report[c]['improves_original'] for c in limits)
report['final_acceptance']='unproven: full categories and video review still required'
(root/'diagnostic_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
