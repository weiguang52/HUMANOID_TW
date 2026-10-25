from pathlib import Path
import json,html,subprocess
import numpy as np
R=Path('/root/gpufree-data/datasets/practice9/repeatability_20261025')
for model in ['original','baseline']:
 p=R/model
 if not (p/'status').exists() or (p/'status').read_text().strip()!='completed':continue
 rows={};encoding=[]
 for job in json.loads((p/'plan.json').read_text()):
  for spec in json.loads(Path(job['jobs']).read_text()):
   f=Path(spec['output']);d=json.loads(f.read_text());a=np.load(f.with_suffix('.npz'))
   xy=a['anchor_actual'][:,:2];delta=xy-xy[0]
   settled=xy[50:]-xy[50] if len(xy)>50 else delta
   key=job['name']+'/'+f.stem
   rows[key]={'net_xy_after_1s_mm':float(np.linalg.norm(settled[-1])*1000),'seconds':d['seconds'],'terminations':d['termination_counts'],'net_xy_mm':float(np.linalg.norm(delta[-1])*1000),'excursion_mm':float(np.linalg.norm(delta,axis=-1).max()*1000),'metrics':d['motion_metrics']}
   meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,nb_read_frames,r_frame_rate','-of','json',str(f.with_suffix('.mp4'))],text=True))['streams'][0]
   assert meta['codec_name']=='h264' and meta['r_frame_rate']=='25/1'
   assert abs(int(meta['nb_read_frames'])-round(d['seconds']*25))<=1
   encoding.append({'file':str(f.with_suffix('.mp4').relative_to(p)),**meta})
 comparisons=[]
 for seed in [42,123,2026]:
  for mid in [2,3]:
   base=f'batch_{seed}/motion_{mid}'
   targets=[f'single_{seed}_{mid}/motion_{mid}']
   if seed==42:targets += [f'repeat_42/motion_{mid}',f'reverse_42/motion_{mid}',f'front_42/motion_{mid}']
   for target in targets:
    a=rows[base];b=rows[target]
    comparisons.append({'reference':base,'target':target,'same_termination':a['terminations']==b['terminations'],'seconds_delta':b['seconds']-a['seconds'],'net_drift_delta_mm':b['net_xy_mm']-a['net_xy_mm'],'max_metric_abs_difference':max(abs(a['metrics'][k]-b['metrics'][k]) for k in a['metrics'])})
 result={'rows':rows,'comparisons':comparisons,'encoding':encoding,'scope':'Controlled diagnostic disables main randomization; seeds do not test randomized robustness. net_xy_mm starts at sample0; net_xy_after_1s_mm matches prior reports.'}
 (p/'summary.json').write_text(json.dumps(result,indent=2))
 page='<meta charset="utf-8"><title>Diagnostic repeatability</title><style>body{background:#181a1e;color:#eee;font:17px sans-serif;margin:30px}a{color:#8cf}video{width:46%}td,th{padding:8px;border:1px solid #777}table{border-collapse:collapse}</style>'
 page+=f'<h1>{model}: diagnostic repeatability</h1><p>Frozen checkpoint; no training. Diagnostic mode disables main randomization, so seeds are not a robustness test. Net XY table uses sample0; summary also includes after-1s drift matching previous reports. Synthetic static/upper diagnostic references, not a new HumanML3D locomotion evaluation. Front camera uses fixed initial direction; does not track heading. Early termination preserved.</p><a href="summary.json">Metrics, comparisons and encoding audit</a>'
 page+='<table><tr><th>Run / motion</th><th>Seconds</th><th>Net XY mm</th><th>Terminations</th><th>Video</th></tr>'
 for k,r in rows.items():page+=f'<tr><td>{k}</td><td>{r["seconds"]:.2f}</td><td>{r["net_xy_mm"]:.2f}</td><td>{html.escape(str(r["terminations"]))}</td><td><a href="{k}.mp4">Replay</a></td></tr>'
 page+='</table>'
 for mid in [2,3]:page+=f'<h2>Motion {mid}: side / front, seed42</h2><video controls src="batch_42/motion_{mid}.mp4"></video><video controls src="front_42/motion_{mid}.mp4"></video>'
 page+='<h2>Joint colors</h2>'
 for row in json.loads(Path('/root/gpufree-data/datasets/practice9/head_locked_20261024/robot_colored.colors.json').read_text()):page+=f'<p><span style="color:{row["hex"]}">&#9632;</span> {html.escape(row["joint"])}</p>'
 (p/'index.html').write_text(page)
 print(model,len(rows),'reports',len(encoding),'videos')
