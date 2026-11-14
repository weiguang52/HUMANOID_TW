"""Standalone scientific plots for issue19; five training-seed bootstrap intervals."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
COLORS={'sym_wrap':'#087f8c','sym_cont':'#d1495b','control_int':'#636363','sym_initial_strict':'#dda448'}
def main():
 g=json.loads((ROOT/'group_statistics.json').read_text());out=ROOT/'figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'figure.dpi':130,'savefig.bbox':'tight'})
 manifests=[]
 def save(fig,name):
  fig.tight_layout()
  for ext in ['png','pdf']:
   p=out/(name+'.'+ext);fig.savefig(p);manifests.append(str(p))
  plt.close(fig)
 def draw(ax,protocol,commands,metric,ylabel,groups=None):
  groups=groups or list(COLORS)
  for j,group in enumerate(groups):
   x=np.arange(len(commands))+(j-(len(groups)-1)/2)*.18
   for i,cmd in enumerate(commands):
    m=g.get(':'.join([protocol,group,cmd]),{}).get(metric)
    if not m or m['mean'] is None or m['ci95'] is None:continue
    y=m['mean'];lo,hi=m['ci95'];ax.errorbar(x[i],y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=3,color=COLORS[group],label=group if i==0 else None)
  ax.set_xticks(np.arange(len(commands)),commands,rotation=10);ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig,ax=plt.subplots(figsize=(10,4.8))
 draw(ax,'clean',['stand','forward_03','forward_05','forward_08'],'full20s_survivor_heading_mean_abs_degrees','Absolute 20s yaw drift (degrees; no-reset survivors)')
 ax.axhline(30,color='black',linestyle='--',label='30 degrees');ax.set_title('Clean zero-yaw heading; all falls and censored trials reported separately')
 save(fig,'zero_yaw_heading')
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 draw(axes[0,0],'clean',['forward_05','forward_08'],'full_xy_error','XY velocity error (m/s)')
 draw(axes[0,1],'clean',['forward_05','forward_08'],'full_vx_mean','Actual forward speed (m/s)')
 draw(axes[1,0],'clean',['yaw_positive','yaw_negative'],'full_yaw_error','Yaw rate error (rad/s)')
 draw(axes[1,1],'combined',['forward_03','forward_05','forward_08'],'fall_rate','Fall probability',groups=['sym_wrap','sym_cont','control_int'])
 axes[1,1].axhline(.01,color='black',linestyle='--');fig.suptitle('Tracking and bounded-combined robustness; 95% seed intervals')
 save(fig,'tracking_and_robustness')
 fig,axes=plt.subplots(1,3,figsize=(16,4.8))
 draw(axes[0],'clean',['forward_05','forward_08'],'startup_signed_degrees_mean','0-2s signed startup offset (degrees)')
 draw(axes[1],'clean',['forward_05','forward_08'],'late_full20s_slope_degrees_per_second_mean','5-20s OLS slope (degrees/s; full survivors)')
 draw(axes[2],'clean',['forward_05','forward_08'],'complete_cycle_sum_mean_degrees','Left+right yaw (degrees/complete initial cycle)')
 for ax in axes:ax.axhline(0,color='black',linestyle='--')
 fig.suptitle('Startup and continuing drift are distinct measurements')
 save(fig,'startup_slope_and_halfcycle')
 fig,axes=plt.subplots(1,2,figsize=(12,4.8))
 draw(axes[0],'clean',['stand','forward_03','forward_05','forward_08'],'fall_rate','Clean fall probability')
 draw(axes[1],'clean',['stand','forward_03','forward_05','forward_08'],'initial_prefix_mean_duration_s','Mean initial episode exposure (s)')
 save(fig,'clean_falls_and_exposure')
 (ROOT/'figure_manifest.json').write_text(json.dumps({'files':manifests,'bootstrap_unit':'five training seeds; exact3125 resamples; evaluation seeds averaged within training seed'},indent=2))
 print('ISSUE19_FIGURES_COMPLETE',len(manifests),flush=True)
if __name__=='__main__':main()
