from pathlib import Path
import json,html,subprocess
import numpy as np
R=Path('/root/gpufree-data/datasets/practice9/perturbation_20261025')
for model in ['original','baseline']:
 p=R/model
 if not (p/'status').exists() or (p/'status').read_text().strip()!='completed':continue
 rows={};encoding=[]
 for job in json.loads((p/'plan.json').read_text()):
  for spec in json.loads(Path(job['jobs']).read_text()):
   f=Path(spec['output']);d=json.loads(f.read_text());a=np.load(f.with_suffix('.npz'));xy=a['anchor_actual'][:,:2]
   kicks=d['perturbations']
   if d['steps']>250:
    assert len(kicks)==1 and kicks[0]['max_write_error']<=1e-6
    assert abs(np.linalg.norm(kicks[0]['requested_delta_xy'])-spec['push_speed'])<1e-6
   else:assert not kicks
   drift=xy-xy[min(50,len(xy)-1)];key=job['name']+'/'+f.stem
   rows[key]={'seconds':d['seconds'],'completed_20s':d['steps']==1000 and not any(d['termination_counts'].values()),'net_xy_after_1s_mm':float(np.linalg.norm(drift[-1])*1000),'max_excursion_mm':float(np.linalg.norm(drift,axis=-1).max()*1000),'terminations':d['termination_counts'],'perturbations':kicks,'metrics':d['motion_metrics'],'push_speed':spec['push_speed'],'seed':job['seed'],'motion_id':spec['motion_id']}
   meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,nb_read_frames,r_frame_rate','-of','json',str(f.with_suffix('.mp4'))],text=True))['streams'][0]
   assert meta['codec_name']=='h264' and meta['r_frame_rate']=='25/1' and abs(int(meta['nb_read_frames'])-round(d['seconds']*25))<=1
   encoding.append({'file':str(f.with_suffix('.mp4').relative_to(p)),**meta})
 comparisons=[]
 for mid in [2,3]:
  a=rows[f'batch_42/motion_{mid}_v05'];b=rows[f'repeat_42/motion_{mid}_v05']
  comparisons.append({'motion_id':mid,'seconds_delta':b['seconds']-a['seconds'],'drift_delta_mm':b['net_xy_after_1s_mm']-a['net_xy_after_1s_mm'],'same_termination':a['terminations']==b['terminations']})
 (p/'summary.json').write_text(json.dumps({'rows':rows,'repeat_checks':comparisons,'encoding':encoding,'scope':'Root velocity kicks only; other randomization disabled. Terminated runs are not20s drift samples.'},indent=2))
 page='<meta charset="utf-8"><title>Audited perturbation</title><style>body{background:#181a1e;color:#eee;font:17px sans-serif;margin:30px}a{color:#8cf}video{width:46%}td,th{padding:8px;border:1px solid #777}table{border-collapse:collapse}</style>'
 page+=f'<h1>{model}: actual velocity perturbation</h1><p>At5s add world XY root velocity0/0.05/0.10m/s with seeded direction. Physics readback checked, not a force pulse. Other randomization disabled. Fixed initial-view cameras; do not follow heading. Synthetic diagnostic references, not HumanML3D source replays.</p><a href="summary.json">Full metrics and injection audit</a><table><tr><th>Run</th><th>Seconds</th><th>After1s drift mm</th><th>Termination</th><th>Video</th></tr>'
 for k,v in rows.items():page+=f'<tr><td>{k}</td><td>{v["seconds"]:.2f}</td><td>{v["net_xy_after_1s_mm"]:.2f}</td><td>{html.escape(str(v["terminations"]))}</td><td><a href="{k}.mp4">Replay</a></td></tr>'
 page+='</table>'
 for speed in [0,5,10]:
  for mid in [2,3]:page+=f'<h2>Motion{mid}, deltaV{speed/100:.2f}m/s: side / front seed42</h2><video controls src="batch_42/motion_{mid}_v{speed:02d}.mp4"></video><video controls src="front_42/motion_{mid}_v{speed:02d}.mp4"></video>'
 page+='<h2>Joint colors</h2>'
 for row in json.loads(Path('/root/gpufree-data/datasets/practice9/head_locked_20261024/robot_colored.colors.json').read_text()):page+=f'<p><span style="color:{row["hex"]}">&#9632;</span> {html.escape(row["joint"])}</p>'
 (p/'index.html').write_text(page);print(model,len(rows),'verified reports/videos')
