"""Matched TW202 review. Keep incomplete episodes and synthetic goals explicit."""
from pathlib import Path
import json, numpy as np, html, hashlib
S=Path('/root/gpufree-data/datasets/practice9/tw202_v1')
R=Path('validation_artifacts/tw202_flat_walk');O=Path('validation_artifacts/tw202_review');O.mkdir(exist_ok=True)
variants=['baseline','precision','neck','root'];results={}
for v in variants:
 c=json.loads((R/v/'comparison.json').read_text());sole=json.loads((R/v/'sole_diagnostics.json').read_text());rows=[]
 for r in c['rows']:
  a=np.load(S/'validation'/v/(r['replay']+'.npz'));end=int(np.flatnonzero(a['terminal'])[0]) if a['terminal'].any() else len(a['terminal']);xy=a['anchor_actual'][50:end,:2];ref=a['anchor_reference'][50:end,:2]
  d=dict(replay=r['replay'],category=r['category'],seconds=end*.02,clean=r['clean_motion_end'],termination=r['termination_counts'])
  d['world_xy_rmse_mm']=float(np.sqrt(np.mean(np.sum((xy-ref)**2,axis=-1)))*1000) if len(xy) else None
  d['net_xy_mm']=float(np.linalg.norm(xy[-1]-xy[0])*1000) if len(xy) else None
  d['reference_net_xy_mm']=float(np.linalg.norm(ref[-1]-ref[0])*1000) if len(ref) else None
  d['five_second_windows']=[]
  for start in range(50,end-250,250):
   x=a['anchor_actual'][start:start+251,:2];rr=a['anchor_reference'][start:start+251,:2]
   d['five_second_windows'].append(dict(start_s=start*.02,xy_rmse_mm=float(np.sqrt(np.mean(np.sum((x-rr)**2,axis=-1)))*1000),net_xy_mm=float(np.linalg.norm(x[-1]-x[0])*1000),excursion_mm=float(np.linalg.norm(x-x[0],axis=-1).max()*1000)))
  rows.append(d)
 cats={r['replay']:r['category'] for r in c['rows']};summary={}
 for cat,record in c['categories'].items():
  rr=[r for r in sole if cats[r['replay']]==cat];xy=[r['world_xy_rmse_mm'] for r in rows if r['category']==cat and r['world_xy_rmse_mm'] is not None]
  d=dict(record);d['sole_replays_with_samples']=sum(not r.get('insufficient_samples',False) for r in rr);d['world_xy_rmse_mm']=float(np.mean(xy)) if xy else None
  for key in ['known_swing_sole_height_m','known_swing_contact','known_stance_missing_contact','contact_lowest_corner_speed_candidate_min_m_s']:
   values=[r[key] for r in rr if r.get(key) is not None];d[key]=float(np.mean(values)) if values else None
  neck=d['metrics']['neck_reference_rmse_rad'];d['neck_mean_axis_rmse_deg']=float(np.rad2deg(np.mean(neck))) if neck is not None else None;summary[cat]=d
 results[v]=dict(categories=summary,replays=rows,synthetic=json.loads((R/v/'diagnostic_summary.json').read_text()))
(O/'comparison.json').write_text(json.dumps(results,indent=2))
for v in variants:
 for cat in ['flat_forward','standing_upper']:
  d=results[v]['categories'][cat];print(v,cat,'clean',d['clean'],'xy',d['world_xy_rmse_mm'],'neck_deg',d['neck_mean_axis_rmse_deg'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(14,8))
plots=[('flat_forward','neck_mean_axis_rmse_deg','Walk: neck mean-axis RMSE (deg)',1),('standing_upper','world_xy_rmse_mm','Stand: world XY RMSE (mm)',1),('flat_forward','known_swing_contact','Walk: known-swing contact (%)',100),('standing_upper','neck_mean_axis_rmse_deg','Stand: neck mean-axis RMSE (deg)',1),('flat_forward','known_stance_missing_contact','Walk: known-stance unloaded (%)',100),('flat_forward','known_swing_sole_height_m','Walk: known-swing sole height (mm)',1000)]
for ax,(cat,key,title,scale) in zip(axes.flat,plots):
 vals=[results[v]['categories'][cat][key]*scale for v in variants];ax.bar(variants,vals,color=['#6b7280','#ef9840','#3482b9','#55a77e']);ax.set_title(title);ax.bar_label(ax.containers[0],fmt='%.2f');ax.margins(y=.18);ax.grid(axis='y',alpha=.2)
fig.suptitle('TW202: one training seed, three replay seeds; lower is better except sole height')
fig.tight_layout();fig.savefig(O/'comparison.png',dpi=140);plt.close(fig)
parts=['<!doctype html><meta charset="utf-8"><title>TW202 review</title><style>body{max-width:1450px;margin:auto;padding:24px;font:17px sans-serif;line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}img{max-width:100%}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%}pre{white-space:pre-wrap}</style><h1>TW202 四组训练效果审查</h1>']
if (O/'findings.html').exists():parts.append((O/'findings.html').read_text())
parts.append('<img src="comparison.png"><h2>完成情况</h2><p>完成参考周期不等于无滑移。站立限位诊断000346、跑步机和非平地样本单独列出。</p><table><tr><th>组</th><th>平地走路</th><th>站立上肢</th><th>转向</th><th>站立限位诊断</th><th>跑步机</th><th>非平地</th></tr>')
for v in variants:
 cells=[v]+[str(results[v]['categories'][c]['clean'])+'/'+str(results[v]['categories'][c]['total']) for c in ['flat_forward','standing_upper','turning','standing_upper_reference_limit_diagnostic','treadmill','nonplanar_review']];parts.append('<tr>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
parts.append('</table><h2>位置、头颈和骨盆指标</h2><table><tr><th>组</th><th>类别</th><th>世界XY RMSE mm</th><th>头颈三轴平均RMSE °</th><th>骨盆yaw RMSE °</th><th>手腕误差 mm</th></tr>')
for v in variants:
 for cat in ['flat_forward','standing_upper']:
  r=results[v]['categories'][cat];m=r['metrics'];cells=[v,cat,f"{r['world_xy_rmse_mm']:.2f}",f"{r['neck_mean_axis_rmse_deg']:.2f}",f"{m['pelvis_heading_rmse_deg']:.2f}",f"{m['wrist_error_m']*1000:.2f}"];parts.append('<tr>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
parts.append('</table><h2>合成固定腿部诊断</h2><p>排除首秒；不同终止时间不可直接用净位移排名。此项是分布外压力测试，不是完整人体动作。</p><table><tr><th>组</th><th>目标</th><th>时长s</th><th>净位移mm</th><th>最大偏移mm</th><th>终止</th></tr>')
for v in variants:
 for r in results[v]['synthetic']:
  cells=[v,r['case'],f"{r['seconds']:.2f}",f"{r['net_xy_mm']:.2f}",f"{r['excursion_mm']:.2f}",str({k:n for k,n in r['termination_counts'].items() if n})];parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
parts.append('</table>')
for clip in ['000832','000016','000211']:
 parts.append('<h2>'+clip+'</h2><p>原始NPY为20fps；机器人为50fps参考时间。两者不按播放时间同步。机器人视频失败后可能重置，不能把后续片段视为连续成功。</p><a href="../tw202_flat_walk/baseline/humanml_source/'+clip+'_source.mp4">原始NPY视频</a><img src="'+clip+'_sheet.jpg"><div class="videos">')
 for v in variants:parts.append('<div>'+v+'<video controls preload="none" src="../tw202_flat_walk/'+v+'/videos/'+clip+'.mp4"></video></div>')
 parts.append('</div>')
parts.append('<h2>完整交付</h2>')
for v in variants:parts.append('<p><a href="../tw202_flat_walk/'+v+'/index.html">'+v+' 完整报告、最终权重目录、所有动作与NPY视频</a></p>')
(O/'index.html').write_text('\n'.join(parts))
