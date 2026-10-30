"""Build a local-only review index after numeric and encoding/parity gates; never promote."""
from pathlib import Path
import json,sys,html,os,re
state=Path(sys.argv[1]);base=Path(__file__).parent;videos=state/'videos'
report=json.loads((base/'two_training_seed_report.json').read_text());assert report['complete'] and report['both_training_seeds_numeric_pass']
audit=json.loads((videos/'delivery_video_audit.json').read_text());assert audit['complete'] and audit['all_encoding_and_replay_checks_pass'] and audit['robot_videos']==46
source=json.loads((videos/'source_video_audit.json').read_text());assert source['complete'] and len(source['rows'])==21
legend=Path('/root/gpufree-data/datasets/practice9/head_locked_20261024/robot_colored.colors.json');colors=json.loads(legend.read_text());assert len(colors)==30
esc=lambda value:html.escape(str(value),quote=True)
def link(path):
 path=Path(path);assert path.is_file(),str(path)
 return esc('/'+path.resolve().relative_to(Path('/root/gpufree-data')).as_posix())
def player(label,path):return '<figure><figcaption>'+esc(label)+'</figcaption><video controls preload="metadata" src="'+link(path)+'"></video></figure>'
plan=json.loads((state/'plan.json').read_text());seed=plan['seed']
result=next(x for x in report['results'] if x['training_seed']==seed)
flat=result['flat']['summary'];fidelity=result['fidelity_all24']
body_rect={m:result['body_jitter'][m]['rectangular'] for m in ('accepted','candidate')}
body_hann={m:result['body_jitter'][m]['hann'] for m in ('accepted','candidate')}
comparison=[]
for label,key,metric_source,digits in [
 ('Loaded contact slip (mm/s)','slip_mm_s',flat,3),
 ('Moving contact slip (mm/s)','moving_slip_mm_s',flat,3),
 ('Fixed leg HF error (deg)','hf_deg',flat,6),
 ('All30 HF error rectangular (deg)','all30',body_rect,6),
 ('Arms10 HF error rectangular (deg)','arms10',body_rect,6),
 ('All30 HF error Hann - diagnostic (deg)','all30',body_hann,6),
 ('Anchor XY path (m)','actual_path_m',fidelity,4),
 ('Reference direction progress ratio','reference_direction_progress_ratio',fidelity,4),
 ('Anchor XY tracking error (mm)','mean_xy_tracking_error_mm',fidelity,3),
 ('Moving XY velocity vector RMSE (m/s)','moving_xy_velocity_vector_rmse_m_s',fidelity,5)]:
 comparison.append('<tr><th>'+esc(label)+'</th><td>'+format(metric_source['accepted'][key],'.'+str(digits)+'f')+'</td><td>'+format(metric_source['candidate'][key],'.'+str(digits)+'f')+'</td></tr>')
quality='<section><h2>Matched24 flat trials - training seed '+esc(seed)+'</h2><table><thead><tr><th>Mean metric</th><th>Accepted policy</th><th>Candidate</th></tr></thead><tbody>'+''.join(comparison)+'</tbody></table><p>All24 pairs complete with identical durations. Reduced contact slip may accompany shorter travel. Review trajectory fidelity and full-body/source videos together; slip and pass counts alone do not establish acceptance. Independent source motions have been repeatedly reviewed, and are not a fresh blind test.</p></section>'
metrics={group:json.loads((state/phase/'metrics.json').read_text())['rows'] for group,phase in [('validation','evaluation'),('independent','independent')]}
sections=[];entries=[]
for group in ('validation','independent'):
 manifest=json.loads((videos/group/'manifest.json').read_text())['motions'];jobs={view:json.loads((videos/group/view/'jobs42.json').read_text()) for view in ('front','side')}
 for i,motion in enumerate(manifest):
  cid=motion['id'];original=next(x for x in source['rows'] if x['group']==group and x['id']==cid);paths={view:Path(next(j['output'] for j in jobs[view] if j['motion_id']==i)).with_suffix('.mp4') for view in jobs}
  body=''.join(player('Robot '+view,paths[view]) for view in ('front','side'))+player('HumanML3D original '+str(original['fps'])+' fps',original['video'])
  outcome=next(x for x in metrics[group] if x['model']=='candidate' and x['seed']==42 and x['id']==cid)
  status='Completed' if outcome['clean'] else 'Early termination; this capability is not established'
  note=esc(status+' | first episode '+format(outcome['first_episode_seconds'],'.2f')+'s | '+outcome['category'])
  sections.append('<section><h2>'+esc(group+' / '+cid)+'</h2><p>'+note+'</p><div class="clips">'+body+'</div><p>Source and robot are independently timed, not phase-synchronized.</p></section>')
  entries.append({'group':group,'id':cid,'robot':{v:str(p) for v,p in paths.items()},'source':original})
template=next(x for x in source['rows'] if x['id']=='000016')
for cid in ('static','upper'):
 paths={}
 for view in ('front','side'):
  jobs=json.loads((videos/'retention'/view/'jobs.json').read_text());job=next(j for j in jobs if Path(j['output']).stem==cid);paths[view]=Path(job['output']).with_suffix('.mp4')
 body=''.join(player('Robot '+view,paths[view]) for view in ('front','side'))+player('000016 construction template; NOT the same synthetic motion',template['video'])
 sections.append('<section><h2>Synthetic 20s / '+cid+'</h2><div class="clips">'+body+'</div><p>Modified synthetic reference, teacher fallback throughout. Template source is not a synchronized counterpart or original contact ground truth.</p></section>')
 entries.append({'group':'retention','id':cid,'robot':{v:str(p) for v,p in paths.items()},'template_source':template,'same_source_motion':False})
items=[]
for row in colors:
 assert re.fullmatch(r'#[0-9a-fA-F]{6}',row['hex'])
 items.append('<li><span style="background:'+row['hex']+'"></span>'+esc(row['joint'])+' / '+esc(row['link'])+'</li>')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TW-RL robot/source review</title><style>body{font:16px system-ui;margin:24px auto;max-width:1600px;padding:0 20px;background:#17191c;color:#eee}section{border-top:1px solid #444;padding:18px 0}.clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}figure{margin:0}video{width:100%;background:#333}figcaption{padding:8px 0}table{border-collapse:collapse;width:100%;max-width:900px}th,td{text-align:left;padding:8px;border-bottom:1px solid #444}.banner{padding:14px;background:#493b17}a{color:#8cceff}ul{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));list-style:none;padding:0;gap:8px}li span{display:inline-block;width:18px;height:18px;margin-right:8px}</style><h1>TW-RL robot and original source review</h1><p class="banner">Numeric gates and encoding/parity passed. Actual visual inspection and final acceptance remain pending. No policy promotion.</p><p>Robot camera follows translation at fixed height and initial orientation. It does not continuously follow heading during turns. HumanML3D plays at original source speed. Players are intentionally independent.</p>'''+quality+''.join(sections)+'<h2>Joint / link color legend</h2><ul>'+''.join(items)+'</ul><p><a href="delivery_video_audit.json">Encoding and replay audit</a> | <a href="source_video_audit.json">Original-source audit</a> | <a href="/datasets/practice9/training_speed_blend_20261026/videos/reference_sources/index.html">Repository renderer original-source index (21 clips)</a></p></html>'
startup=json.loads((base/'startup_frame_audit.json').read_text())
if not (startup['complete'] and startup['observed']==92 and not startup['blank_cases']):page=page.replace('<h1>TW-RL robot and original source review</h1>','<h1>TW-RL robot and original source review</h1><p class="banner">Renderer startup-frame checks are pending. Final visual acceptance pending.</p>')
(videos/'index.html').write_text(page)
(videos/'delivery_index.json').write_text(json.dumps({'robot_videos':46,'original_source_videos':21,'entries':entries,'legend':str(legend),'visual_reviewed':False,'final_acceptance':False},indent=2))
print('Review index built: 46 robot videos, 21 original source videos; visual review pending')
