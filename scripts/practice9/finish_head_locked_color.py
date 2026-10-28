"""Assemble matched colored front/side replay delivery after both workers finish."""
from pathlib import Path
import hashlib,html,json,shutil,subprocess
import numpy as np
S=Path('/root/gpufree-data/datasets/practice9/head_locked_20261024')
E=S/'evaluation_colored'
D=Path('validation_artifacts/head_locked_20261024/evaluation_colored/neck_only')
assert (E/'validation/neck_only/status').read_text().strip()=='completed'
assert (E/'front/neck_only/status').read_text().strip()=='completed'
assert (E/'diagnostic_front/status').read_text().strip()=='completed'
shutil.copytree(E/'diagnostic_front/neck_only',D/'diagnostic_front',dirs_exist_ok=True)
old=json.loads((S/'evaluation/validation/neck_only/comparison.json').read_text())
new=json.loads((D/'comparison.json').read_text())
manifest=json.loads((D/'evaluation_manifest.json').read_text())
assert len(new['rows'])==42
comparison={k:{'previous':[old['categories'][k]['clean'],old['categories'][k]['total']], 'colored':[v['clean'],v['total']]} for k,v in new['categories'].items()}
(D/'rerun_comparison.json').write_text(json.dumps(comparison,indent=2))
front=E/'front/neck_only'
shutil.copytree(front/'videos',D/'videos_front',dirs_exist_ok=True)
for p in front.glob('*.json'):
 if p.name.endswith('.video.json'):
  shutil.copy2(p,D/(p.stem+'.front.json'))
for name in ['robot_colored.colors.json','robot_colored.audit.json']:shutil.copy2(S/name,D/name)
camera_audit=[]
for p in front.glob('*.camera.npz'):
 shutil.copy2(p,D/(p.stem+'.front.npz'))
 with np.load(p) as a:
  continuous=~(a['reset'][1:] | a['reset'][:-1])
  delta=np.diff(a['camera_origin'],axis=0)[continuous]
  vertical=float(np.max(np.abs(delta[:,2]))) if len(delta) else 0.
  assert vertical<1.e-6
  camera_audit.append(dict(file=p.name,frames=len(a['reset']),continuous_max_vertical_step_m=vertical))
assert len(camera_audit)==14
(D/'front_camera_audit.json').write_text(json.dumps(camera_audit,indent=2))
differences=[]
for p in (E/'validation/neck_only').glob('*.seed*.json'):
 if p.name.endswith('.summary.json'):continue
 a=json.loads((S/'evaluation/validation/neck_only'/p.name).read_text())
 b=json.loads(p.read_text())
 differences.append(dict(file=p.name,max_metric_difference=max(abs(a['motion_metrics'][k]-v) for k,v in b['motion_metrics'].items())))
assert len(differences)==42
(D/'metric_reproducibility.json').write_text(json.dumps(differences,indent=2))
encoding=[]
for row in manifest['motions']:
 for view in ['videos','videos_front']:
  path=D/view/(row['id']+'.mp4')
  meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,nb_read_frames,r_frame_rate','-of','json',str(path)],text=True))['streams'][0]
  assert meta['codec_name']=='h264' and meta['r_frame_rate']=='50/1'
  assert int(meta['nb_read_frames']) in (row['frames'],row['frames']+1)
  encoding.append(dict(file=str(path.relative_to(D)),**meta))
(D/'dual_view_encoding.json').write_text(json.dumps(encoding,indent=2))
page='''<!doctype html><meta charset="utf-8"><title>Head-locked colored evaluation</title>
<style>body{max-width:1500px;margin:30px auto;background:#17191c;color:#eee;font:17px sans-serif;padding:16px}a{color:#8cf}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #666}.legend{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}i{display:inline-block;width:20px;height:20px;margin-right:8px;vertical-align:middle}@media(max-width:800px){.views,.legend{grid-template-columns:1fr}}</style>
<h1>Head-locked robot: colored front and side evaluation</h1>
<p>Checkpoint model_27995.pt. Neck, neck roll and head pitch remain locked at their initial angles. Colors are baked into a separate USD; mass and physics are unchanged. No runtime coloring.</p>
<p>42 metric replays (14 motions x 3 seeds). Videos use seed42. Front camera follows translation from the initial frontal direction; turns can face away. Source HumanML3D uses its original timing and is not phase-synchronized with robot video. Early terminations and resets are included; see comparison.json.</p>
<p><a href="comparison.json">Full metrics</a> / <a href="rerun_comparison.json">Previous vs colored rerun</a> / <a href="dual_view_encoding.json">Video checks</a></p>
<table><tr><th>Category</th><th>Previous head-locked</th><th>Colored rerun</th></tr>'''
for name,r in comparison.items():page+=f'<tr><td>{html.escape(name)}</td><td>{r["previous"][0]}/{r["previous"][1]}</td><td>{r["colored"][0]}/{r["colored"][1]}</td></tr>'
page+='</table>'
for row in manifest['motions']:
 cid=row['id'];refs=list((D/'humanml_source').glob(cid+'*.mp4'));assert len(refs)==1
 page+=f'<h2>{cid} / {html.escape(row.get("evaluation_class",row["category"]))}</h2><div class="views">'
 for title,url in [('Front',f'videos_front/{cid}.mp4'),('Side',f'videos/{cid}.mp4'),('Original HumanML3D',str(refs[0].relative_to(D)))]:page+=f'<div><h3>{title}</h3><video controls preload="metadata" src="{url}"></video></div>'
 page+='</div>'
page+='<h2>Static diagnostics (20 seconds)</h2><p>Separate diagnostic cases, not held-out motion acceptance.</p>'
for name in ['corrected_static','corrected_upper']:
 page+=f'<h3>{name}</h3><div class="views"><div><h4>Front</h4><video controls preload="metadata" src="diagnostic_front/{name}.mp4"></video></div><div><h4>Side</h4><video controls preload="metadata" src="diagnostic/{name}.mp4"></video></div></div>'
 meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate','-of','json',str(D/'diagnostic_front'/(name+'.mp4'))],text=True))['streams'][0]
 assert meta['codec_name']=='h264' and int(meta['nb_frames'])==500 and meta['r_frame_rate']=='25/1'

page+='<h2>Joint color legend</h2><div class="legend">'
for row in json.loads((D/'robot_colored.colors.json').read_text()):page+=f'<div><i style="background:{row["hex"]}"></i>{html.escape(row["joint"])}</div>'
page+='</div>';(D/'index.html').write_text(page)
sha={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file() and p.name!='sha256.json'}
(D/'sha256.json').write_text(json.dumps(sha,indent=2))
print(json.dumps(comparison,indent=2));print('ASSEMBLED',D)
