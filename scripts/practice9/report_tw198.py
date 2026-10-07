"""Summarize controlled diagnostic runs; exclude first second."""
import json,html,shutil,subprocess
from pathlib import Path
import numpy as np
S=Path('/root/gpufree-data/datasets/practice9/tw198_v1')
O=Path('validation_artifacts/tw198_standing_diagnostic');O.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted(S.glob('*/*/*.npz')):
 a=np.load(p);meta=json.loads(p.with_suffix('.json').read_text());dt=float(a['control_dt']);steps=a['control_step'].astype(int);mask=steps>=50
 if not mask.any():
  rows.append(dict(model=p.parts[-3],rate=p.parts[-2],case=p.stem,seconds=meta['seconds'],termination={k:v for k,v in meta['termination_counts'].items() if v},insufficient_after_1s=True));continue
 anchor=a['anchor_actual'][50:,:2];feet=a['foot_position'][mask,:,:2];target=a['target_q'][mask];ref=a['reference_q'][steps[mask]]
 names=a['joint_names'].tolist();legs=[i for i,n in enumerate(names) if any(t in n for t in ('hip','knee','ankle','foot'))]
 r=dict(model=p.parts[-3],rate=p.parts[-2],case=p.stem,seconds=meta['seconds'],termination={k:v for k,v in meta['termination_counts'].items() if v},anchor_net_mm=float(np.linalg.norm(anchor[-1]-anchor[0])*1000),anchor_excursion_mm=float(np.linalg.norm(anchor-anchor[0],axis=-1).max()*1000),feet_net_mm=(np.linalg.norm(feet[-1]-feet[0],axis=-1)*1000).tolist(),fz_below_1_fraction=(a['force'][mask,:,2]<1).mean(0).tolist(),leg_target_range_deg=np.rad2deg(np.ptp(target[:,legs],axis=0)).tolist(),leg_names=[names[i] for i in legs],target_reference_rmse_rad=float(np.sqrt(np.mean((target-ref)**2))),joint_tracking_rmse_rad=float(np.sqrt(np.mean((a['q'][mask]-target)**2))))
 r["five_second_windows"]=[]
 for start in range(50,len(a["anchor_actual"])-250,250):
  xy=a["anchor_actual"][start:start+251,:2];r["five_second_windows"].append(dict(start_s=start*.02,net_mm=float(np.linalg.norm(xy[-1]-xy[0])*1000),excursion_mm=float(np.linalg.norm(xy-xy[0],axis=-1).max()*1000)))
 rows.append(r)
(O/'metrics.json').write_text(json.dumps(rows,indent=2))
print('runs',len(rows))
for r in rows:print(r['model'],r['rate'],r['case'],r['seconds'],round(r.get('anchor_net_mm',float('nan')),2),r['termination'],round(max(r.get('leg_target_range_deg',[float('nan')])),2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,7))
for mi,model in enumerate(['original','corrected']):
 for rate in ['200hz','400hz']:
  for controller in ['reference_pd','policy']:
   p=S/model/rate/('corrected_static_'+controller+'.npz')
   if not p.exists():continue
   a=np.load(p);t=np.arange(len(a['anchor_actual']))*.02;xy=a['anchor_actual'][:,:2];d=np.linalg.norm(xy-xy[0],axis=-1)*1000
   axes[0,mi].plot(t,d,label=rate+' '+controller)
   ix=[i for i,n in enumerate(a['joint_names']) if any(s in n for s in ['hip','knee','ankle','foot'])]
   q=a['target_q'][:,ix];dev=np.rad2deg(np.max(np.abs(q-q[0]),axis=1))
   axes[1,mi].plot(a['control_step']*.02,dev,label=rate+' '+controller)
 axes[0,mi].set_title(model+' checkpoint / corrected static target')
 for ax in axes[:,mi]:ax.grid(alpha=.25);ax.legend(fontsize=7);ax.set_xlabel('seconds')
axes[0,0].set_ylabel('body XY displacement from reset (mm)');axes[1,0].set_ylabel('max leg target change from reset (deg)')
fig.tight_layout();fig.savefig(O/'comparison.png',dpi=140);plt.close(fig)
for p in S.glob('*/*/*.json'):
 if p.name in ['jobs.json']:continue
 dest=O/p.relative_to(S);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for p in S.glob('*/*/*.mp4'):
 dest=O/p.relative_to(S);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for name in ['manifest.json','geometry.json']:
 shutil.copy2(S/name,O/name)
source=Path('validation_artifacts/tw195_flat_walk/original/humanml_source')
for p in source.glob('000016_source.*'):shutil.copy2(p,O/p.name)
parts=['<!doctype html><meta charset="utf-8"><title>TW198 Standing diagnosis</title><style>body{font:16px sans-serif;max-width:1400px;margin:30px auto;padding:20px}td,th{padding:7px;border:1px solid #ddd}table{border-collapse:collapse}video{width:560px;max-width:100%}img{max-width:100%}</style><h1>TW-198 固定目标 PD / 策略对照</h1>',
'<p>仅诊断，未启动训练。000016 第100帧生成静止目标；另设腿/腰/根固定、1秒后仅上肢表达的合成目标。原始人体接触时序未修改，合成双支撑标签仅用于受控测试。每组最多20秒，seed42、相同初态，无观测/初态噪声，μ=1、恢复系数0，控制50Hz；物理200/400Hz。禁用手部跟踪终止，保留高度/倾角终止。</p>',
'<p>Displacement excludes first second; early failures retained. Synthetic out-of-distribution test, not full-motion acceptance. Foot displacement is not measured contact-point slip.</p>',
'<img src="comparison.png"><table><tr><th>权重</th><th>物理频率</th><th>目标/控制器</th><th>持续秒</th><th>机身净位移mm</th><th>终止</th><th>腿目标最大变化幅度°</th></tr>']
for r in rows:
 parts.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['model'],r['rate'],r['case'],r['seconds'],round(r.get('anchor_net_mm',float('nan')),3),r['termination'],round(max(r.get('leg_target_range_deg',[float('nan')])),2)])+'</tr>')
parts+=['</table><h2>Original NPY (20fps)</h2><p>Synthetic targets freeze the legs/root; not synchronized with source.</p><video controls src="000016_source.mp4"></video><h2>Fixed world camera, 25fps realtime</h2>']
for p in sorted(O.glob('*/*/*.mp4')):parts.append('<h3>'+str(p.relative_to(O))+'</h3><video controls src="'+str(p.relative_to(O))+'"></video>')
parts.insert(3,'<h2>Same corrected target: fixed PD vs trained policy</h2><video controls src="pd_vs_policy.mp4" style="width:1100px"></video>')
parts.insert(3,(O/'findings.html').read_text() if (O/'findings.html').exists() else '')
(O/'index.html').write_text('\n'.join(parts))
import hashlib
provenance=[]
for p in sorted(S.glob('*/*/*.npz')):
 provenance.append(dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(O/'telemetry_provenance.json').write_text(json.dumps(provenance,indent=2))
media=[]
for p in sorted(O.rglob('*.mp4')):
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate','-of','json',str(p)],text=True))['streams'][0]
 assert info['codec_name']=='h264' and int(info['nb_frames'])>0
 if p.name=='pd_vs_policy.mp4':assert int(info['nb_frames'])==500 and info['r_frame_rate']=='25/1'
 elif p.name!='000016_source.mp4':
  meta=json.loads(p.with_suffix('.json').read_text());assert int(info['nb_frames'])==(meta['steps']+1)//2;assert info['r_frame_rate']=='25/1'
 media.append(dict(file=str(p.relative_to(O)),**info))
(O/'media_audit.json').write_text(json.dumps(media,indent=2))
print('media verified',len(media))
