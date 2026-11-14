"""Seed-level intervals for paired phase/USD diagnostic, preserving missing cells."""
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from issue20_common import ROOT,save,sha
COMMANDS=['stand','forward_03','forward_05','forward_08']
LABELS=['stand','0.3 m/s','0.5 m/s','0.8 m/s']
def error(ax,x,values,color,label,offset=0):
 means=np.array([np.nan if v['mean'] is None else v['mean'] for v in values]);lo=np.array([np.nan if v['ci95'] is None else v['ci95'][0] for v in values]);hi=np.array([np.nan if v['ci95'] is None else v['ci95'][1] for v in values]);ax.errorbar(np.asarray(x)+offset,means,yerr=np.maximum(np.stack([means-lo,hi-means]),0),fmt='o-',capsize=4,color=color,label=label,linewidth=1.2)
 for i,v in enumerate(values):ax.scatter(np.full(5,x[i]+offset)+np.linspace(-.035,.035,5),[np.nan if z is None else z for z in v['seeds']],s=18,color=color,alpha=.55)
def main():
 records={u:json.loads((ROOT/(u+'_analysis.json')).read_text()) for u in ['symmetric','original'] if (ROOT/(u+'_analysis.json')).exists()};colors={'symmetric':'#1769aa','original':'#bd4d21'};folder=ROOT/'figures';folder.mkdir(exist_ok=True);files=[]
 def finish(fig,name):
  fig.text(.5,.005,'Dots: five training seeds; bars: individual paired seed-bootstrap95 intervals (3125 resamples). Evaluation seeds averaged within training seed.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.035,1,1))
  for suffix in ['png','pdf']:
   p=folder/(name+'.'+suffix);fig.savefig(p,dpi=170);files.append(str(p))
  plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.8))
 for i,(usd,a) in enumerate(records.items()):
  for ax,key in zip(axes,['S20_degrees','D20_degrees']):error(ax,np.arange(4),[a['paired_group_statistics'][c][key] for c in COMMANDS],colors[usd],usd,offset=(i-.5)*.11)
 for ax in axes:ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(range(4),LABELS);ax.set_ylabel('degrees');ax.grid(alpha=.2);ax.legend()
 axes[0].axhspan(-5,5,color='green',alpha=.08,label='S +/-5deg gate');axes[0].set_title('Mirror-asymmetric component S = yaw(k0)+yaw(k10)');axes[1].set_title('Equivariant component D = (yaw(k0)-yaw(k10))/2');finish(fig,'paired_S_D')
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 keys=['signed20_degrees','absolute20_degrees','startup_signed_degrees_mean','late_full20s_slope_degrees_per_second_mean'];titles=['20s signed heading drift (deg)','20s absolute heading drift (deg)','0-2s signed startup offset (deg)','5-20s signed OLS slope (deg/s)']
 for ax,key,title in zip(axes.ravel(),keys,titles):
  for j,(usd,a) in enumerate(records.items()):
   for k,phase in enumerate([0,10]):
    vals=[a['cell_statistics'][f'{phase}:{c}'][key] for c in COMMANDS];error(ax,np.arange(4),vals,colors[usd] if phase==0 else ('#39a7b8' if usd=='symmetric' else '#dba346'),f'{usd} k={phase}',offset=(j*2+k-1.5)*.07)
  ax.set_title(title);ax.set_xticks(range(4),LABELS);ax.axhline(0,color='black',linewidth=.7);ax.grid(alpha=.2);ax.legend(fontsize=8)
 finish(fig,'heading_startup_and_slope')
 fig,axes=plt.subplots(1,2,figsize=(12,4.8))
 for j,(usd,a) in enumerate(records.items()):
  for ax,key in zip(axes,['half_swap_left_residual_degrees','half_swap_right_residual_degrees']):error(ax,np.arange(4),[a['paired_group_statistics'][c][key] for c in COMMANDS],colors[usd],usd,offset=(j-.5)*.1)
 for ax,title in zip(axes,['k10 left + k0 right','k10 right + k0 left']):ax.set_title(title+' (deg/complete cycle)');ax.set_xticks(range(4),LABELS);ax.axhline(0,color='black',linewidth=.7);ax.grid(alpha=.2);ax.legend()
 finish(fig,'halfcycle_swap_residuals')
 fig,axes=plt.subplots(1,2,figsize=(12,4.8))
 for j,(usd,a) in enumerate(records.items()):
  for k,phase in enumerate([0,10]):
   for ax,key,scale in [(axes[0],'fall_rate',100),(axes[1],'initial_prefix_mean_duration_s',1)]:
    vals=[]
    for c in COMMANDS:
     v=a['cell_statistics'][f'{phase}:{c}'][key];vals.append({'mean':None if v['mean'] is None else v['mean']*scale,'ci95':None if v['ci95'] is None else [x*scale for x in v['ci95']],'seeds':[None if x is None else x*scale for x in v['seeds']]})
    error(ax,np.arange(4),vals,colors[usd] if phase==0 else ('#39a7b8' if usd=='symmetric' else '#dba346'),f'{usd} k={phase}',offset=(j*2+k-1.5)*.07)
 for ax,title in zip(axes,['Falls (% initial trials)','Initial-episode exposure (seconds)']):ax.set_title(title);ax.set_xticks(range(4),LABELS);ax.grid(alpha=.2);ax.legend(fontsize=8)
 finish(fig,'falls_and_exposure');save('figure_manifest.json',{'files':files,'file_sha256':{p:sha(p) for p in files},'seed_unit':'five training seeds','manual_review_required':True});print('ISSUE20_FIGURES_COMPLETE',len(files),flush=True)
if __name__=='__main__':main()
