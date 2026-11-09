"""Risk denominators and uncertainty are explicit in all plots."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/gpufree-data/tw_stage1/issue11_20261110');FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)

def rows(name):return list(csv.DictReader((ROOT/name).open()))

def number(v):return np.nan if v in ['',None] else float(v)

def factor_plot(name,factors,title,outfile):
 data=rows(name);fig,axes=plt.subplots(1,len(factors),figsize=(4*len(factors),4.6));axes=np.atleast_1d(axes)
 for ax,factor in zip(axes,factors):
  levels=list(dict.fromkeys(x['level'] for x in data if x['factor']==factor))
  for j,stage in enumerate([1500,4500]):
   rr=[next(x for x in data if x['stage']==str(stage) and x['command']=='forward_03' and x['factor']==factor and x['level']==level) for level in levels]
   y=np.array([number(x['rate']) for x in rr])*100;lo=np.array([number(x['wilson95_low']) for x in rr])*100;hi=np.array([number(x['wilson95_high']) for x in rr])*100
   xx=np.arange(len(levels))+(j-.5)*.15;ax.errorbar(xx,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='o-',capsize=4,label=f'{stage} updates')
  labels=[level+'\n'+str(next(x['n'] for x in data if x['stage']=='1500' and x['command']=='forward_03' and x['factor']==factor and x['level']==level))+'\n'+str(next(x['n'] for x in data if x['stage']=='4500' and x['command']=='forward_03' and x['factor']==factor and x['level']==level)) for level in levels]
  ax.set_xticks(range(len(levels)),labels,fontsize=8);ax.set_xlabel('level / n1500 / n4500');ax.set_title(factor);ax.set_ylabel('risk (%) + descriptive Wilson95%');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle(title);fig.tight_layout();fig.savefig(FIG/outfile,dpi=160);plt.close(fig)

factor_plot('trial_stratified_risks.csv',['static_q','mass_q','delay_ms'],'0.3m/s:20s FIRST-FAILURE risk; all initial trials, not case fractions','baseline_trial_risks.png')
factor_plot('push_stratified_risks.csv',['amplitude_q','direction','support_code','phase_bin'],'0.3m/s:1s post-push risk; at-risk full-window pushes only','push_window_risks.png')
data=rows('trial_joint_strata.csv');vmax=max(number(x['rate']) for x in data if x['command']=='forward_03');fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
for row,stage in enumerate([1500,4500]):
 for col,delay in enumerate([0,20]):
  ax=axes[row,col];mat=np.full((4,4),np.nan)
  for x in data:
   if x['stage']==str(stage) and x['command']=='forward_03' and x['delay_ms']==str(delay):
    i=int(x['static_q'])-1;j=int(x['mass_q'])-1;mat[i,j]=number(x['rate'])*100;ax.text(j,i,f"{x['falls']}/{x['n']}",ha='center',va='center',fontsize=8)
  im=ax.imshow(mat,vmin=0,vmax=vmax*100,cmap='YlOrRd');ax.set_xticks(range(4),[1,2,3,4]);ax.set_yticks(range(4),[1,2,3,4]);ax.set_xlabel('base mass quartile');ax.set_ylabel('foot static-friction quartile');ax.set_title(f'{stage}updates; delay{delay}ms')
fig.colorbar(im,ax=axes.ravel().tolist(),label='20s fall risk (%)',shrink=.7);fig.suptitle('0.3m/s joint baseline risks; labels=falls/ALL trials');fig.savefig(FIG/'joint_baseline_heatmap.png',dpi=160);plt.close(fig)
modes=rows('failure_mode_comparison.csv');keys=[x['case_mode'] for x in sorted([x for x in modes if x['stage']=='4500'],key=lambda x:-int(x['failures']))[:6]];fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(keys))
for i,stage in enumerate([1500,4500]):
 values=[sum(int(r['failures']) for r in modes if r['stage']==str(stage) and r['case_mode']==key) for key in keys];ax.bar(x+(i-.5)*.35,values,.35,label=f'{stage}updates')
ax.set_xticks(x,[k.replace('__','\n') for k in keys],fontsize=8);ax.set_ylabel('first failures (CASE COUNTS, not risk)');ax.set_title('Low-speed failure composition; exposure risks are in the risk tables');ax.legend();fig.tight_layout();fig.savefig(FIG/'failure_mode_comparison.png',dpi=160);plt.close(fig)
corners=json.loads((ROOT/'corner_diagnostics.json').read_text())['4500:forward_03'];fig,axes=plt.subplots(1,2,figsize=(11,4.8));labels=['strict','envelope'];x=np.arange(2)
axes[0].bar(x-.18,[100*corners[k]['trial_coverage'] for k in labels],.36,label='eligible corner trial coverage');axes[0].bar(x+.18,[100*(corners[k]['proximal_failure_fraction'] or 0) for k in labels],.36,label='first failures within1s of corner');axes[0].set_xticks(x,labels);axes[0].set_ylabel('%');axes[0].set_title('Prespecified corners,4500updates,0.3m/s');axes[0].legend(fontsize=7)
rr=[corners['original']]+[corners[k]['remaining_trial_risk'] for k in labels];means=np.array([r['rate'] for r in rr])*100;lo=np.array([r['wilson95_low'] for r in rr])*100;hi=np.array([r['wilson95_high'] for r in rr])*100
axes[1].errorbar(range(3),means,yerr=[np.maximum(0,means-lo),np.maximum(0,hi-means)],fmt='o',capsize=5);axes[1].set_xticks(range(3),['all trials','exclude strict','exclude envelope']);axes[1].set_ylabel('20s fall risk (%) + Wilson95%');axes[1].set_title('Descriptive whole-trial exclusion; NOT new acceptance');fig.tight_layout();fig.savefig(FIG/'corner_diagnostics.png',dpi=160);plt.close(fig)
effects=rows('factor_effects.csv');selected=[x for x in effects if x['stage']=='4500' and x['command']=='forward_03' and ((x['unit']=='20s_trial' and x['factor'] in ['static_q','mass_q','delay_ms']) or (x['unit']=='1s_push_window' and x['factor'] in ['amplitude_q','support_code','direction'])) and x['estimable_all5seeds']=='True'];fig,ax=plt.subplots(figsize=(10,8))
for i,r in enumerate(selected):
 m=100*number(r['seed_mean_risk_difference']);lo=100*number(r['difference_seed_boot95_low']);hi=100*number(r['difference_seed_boot95_high']);ax.errorbar(m,i,xerr=[[max(0,m-lo)],[max(0,hi-m)]],fmt='o',capsize=3)
ax.set_yticks(range(len(selected)),[f"{r['unit']}: {r['factor']} {r['target_level']} vs {r['reference_level']}" for r in selected],fontsize=8);ax.axvline(0,color='gray',ls='--');ax.set_xlabel('unadjusted equal-seed risk difference (pp), seed-bootstrap95%');ax.set_title('4500updates,0.3m/s;5training seeds; no causal/multiplicity claim');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(FIG/'factor_seed_effects.png',dpi=160);plt.close(fig)
print('PLOTS_COMPLETE',flush=True)
