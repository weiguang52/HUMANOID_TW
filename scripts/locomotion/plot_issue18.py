"""Standalone figures: equal training-seed weights, exact five-seed intervals."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
GROUPS=['control6000','control_int','sym_int'];MODES=['original','strict'];COLORS=['#667085','#1570EF']
def read(n):return json.loads((ROOT/n).read_text())
def main():
 assert read('raw_analysis_audit.json')['passed'];g=read('group_statistics.json');c=read('classification.json');out=ROOT/'figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
 saved=[]
 def finish(fig,name):
  fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig);saved.append(name)
 def bars(ax,group,cmds,metric,title,ylabel):
  x=np.arange(len(cmds))
  for mode,offset,color in zip(MODES,[-.18,.18],COLORS):
   records=[g[f'{mode}:{group}:{cmd}'][metric] for cmd in cmds]
   for i,r in enumerate(records):
    if r['mean'] is None:ax.text(i+offset,.03,'missing',rotation=90,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8);continue
    v=r['mean'];lo,hi=r['ci95'];ax.bar(i+offset,v,width=.34,color=color,label=mode if i==0 else None);ax.errorbar(i+offset,v,yerr=[[max(v-lo,0)],[max(hi-v,0)]],fmt='none',ecolor='black',capsize=3,lw=1)
  ax.set_xticks(x,[s.replace('forward_','v=').replace('yaw_positive','yaw +.3').replace('yaw_negative','yaw -.3') for s in cmds],rotation=25);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 zero=['stand','forward_03','forward_05','forward_08']
 fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
 for col,group in enumerate(GROUPS):
  bars(axes[0,col],group,zero,'full20s_survivor_heading_mean_abs_degrees',group+' | '+c['cohorts'][group]['classification'],'Mean absolute heading (deg)');axes[0,col].axhline(30,color='#D92D20',ls='--',lw=1)
  bars(axes[1,col],group,zero,'full20s_survivor_heading_signed_mean_degrees',group,'Mean signed heading (deg)');axes[1,col].axhline(0,color='black',lw=.7)
 axes[0,0].legend();fig.suptitle('Strict projection: full 20 s initial-episode survivors only\nFive training seeds; paired evaluation seeds; exact seed bootstrap 95% intervals. Falls/prefixes reported separately.')
 finish(fig,'zero_yaw_heading')
 commands=['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']
 metrics=[('fall_rate','Initial-trial fall fraction'),('initial_prefix_mean_duration_s','Mean initial prefix (s)'),('full_xy_error','XY tracking error (m/s)'),('full_yaw_error','Yaw tracking error (rad/s)'),('full_vx_mean','Mean forward velocity (m/s)')]
 # Metric names are bound to the inherited pooled-statistics schema; fail on a mismatch.
 fig,axes=plt.subplots(5,3,figsize=(15,18),layout='constrained')
 for row,(metric,label) in enumerate(metrics):
  for col,group in enumerate(GROUPS):bars(axes[row,col],group,commands,metric,group if row==0 else '',label)
 axes[0,0].legend();fig.suptitle('All commands: performance and censoring | five training-seed bootstrap 95% intervals')
 finish(fig,'tracking_falls_prefix')
 fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained');parts=['left','right','sum']
 for ax,group in zip(axes,GROUPS):
  for mode,offset,color in zip(MODES,[-.18,.18],COLORS):
   for i,part in enumerate(parts):
    r=g[f'{mode}:{group}:forward_05'][f'complete_cycle_{part}_mean_degrees']
    if r['mean'] is None:ax.text(i+offset,.03,'no complete cycles',rotation=90,transform=ax.get_xaxis_transform(),fontsize=8);continue
    v=r['mean'];lo,hi=r['ci95'];ax.bar(i+offset,v,.34,color=color,label=mode if i==0 else None);ax.errorbar(i+offset,v,yerr=[[max(v-lo,0)],[max(hi-v,0)]],fmt='none',ecolor='black',capsize=3)
  ax.set_xticks(range(3),['idx 0–9','idx 10–19','left + right']);ax.axhline(0,color='black',lw=.8);ax.set_title(group);ax.set_ylabel('Yaw increment per complete cycle (deg)');ax.grid(axis='y',alpha=.2)
 axes[0].legend();fig.suptitle('Forward 0.5: pre-action integer phase, initial-episode complete cycles only\nInterrupted cycles excluded and counted; all valid half totals and exposures remain in tables.')
 finish(fig,'half_cycle_yaw')
 (ROOT/'figure_manifest.json').write_text(json.dumps({'figures':saved,'manual_review_required':True},indent=2)+'\n')
if __name__=='__main__':main()
