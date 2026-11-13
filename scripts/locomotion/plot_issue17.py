"""Source-bound scientific plots; missing full-horizon endpoints remain explicit."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from issue15_math import paired_summary
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113');MAIN=['stand','forward_03','forward_05','forward_08'];ABS='full20s_survivor_heading_mean_abs_degrees';SIGNED='full20s_survivor_heading_signed_mean_degrees'
def load(n):return json.loads((ROOT/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def point(ax,x,value,color):
 if value is None:ax.text(x,.5,'NA: censored',transform=ax.get_xaxis_transform(),ha='center');return
 m=value['mean'];ci=value['bootstrap95'];ax.errorbar(x,m,yerr=np.array([[m-ci[0]],[ci[1]-m]]) if ci else None,fmt='o',color=color,capsize=4)
def main():
 assert (ROOT/'statistics_complete').exists();g=load('group_statistics.json');s=load('seed_statistics.json');p=load('paired_comparisons.json');folder=ROOT/'figures';folder.mkdir(exist_ok=True);outputs={}
 def save(fig,name):
  fig.tight_layout()
  for ext in ['png','svg']:
   f=folder/(name+'.'+ext);fig.savefig(f,dpi=160,bbox_inches='tight');outputs[str(f.relative_to(ROOT))]=sha(f)
  plt.close(fig)
 fig,axes=plt.subplots(1,4,figsize=(18,5))
 for ax,c in zip(axes,MAIN):
  for i,group in enumerate(['control_int','sym_int']):
   point(ax,i,g[f'clean:{group}:{c}'][ABS],f'C{i}')
   for seed in range(1,6):
    v=s[f'clean:{group}:{c}'][str(seed)][ABS]
    if v is not None:ax.plot(i+(seed-3)*.025,v,'.',color=f'C{seed+1}',alpha=.6)
  ax.axhline(30,color='green',ls='--');ax.set_xticks([0,1],['control_int','sym_int']);ax.set_title(c);ax.set_ylabel('mean abs20s heading (deg)');ax.grid(alpha=.2)
 fig.suptitle('Frozen30deg mean gate; intervals are training-seed bootstrap95, reported separately\nNo survivor endpoint invented; paired training seeds are the sample units.')
 save(fig,'heading_acceptance')
 fig,axes=plt.subplots(2,4,figsize=(18,10))
 for row,group in enumerate(['control_int','sym_int']):
  for ax,c in zip(axes[row],MAIN):
   for x,protocol in enumerate(['clean','mirror']):point(ax,x,g[f'{protocol}:{group}:{c}'][SIGNED],f'C{x}')
   for seed in range(1,6):
    v=[s[f'{protocol}:{group}:{c}'][str(seed)][SIGNED] for protocol in ['clean','mirror']]
    if all(x is not None for x in v):ax.plot([0,1],v,'o-',alpha=.35,lw=.6,ms=3)
   ax.axhline(0,color='black',lw=.5);ax.set_xticks([0,1],['original','spatial mirror']);ax.set_title(group+' | '+c);ax.set_ylabel('signed20s heading (deg)')
 fig.suptitle('Pure SPATIAL evaluation reflection keeps scalar clock unchanged\nTraining augmentation instead adds half-cycle phase shift; these are separate interventions.')
 save(fig,'original_mirrored_heading')
 fig,axes=plt.subplots(2,3,figsize=(16,9));entries=[('primary0.5 mean absolute heading',p['primary'])]+[(c+' absolute(O+M)',p['abs_O_plus_M_'+c]['paired_difference']) for c in MAIN]+[('fixed-state action E',p['action_E']['paired_difference'])]
 for ax,(label,value) in zip(axes.flat,entries):
  point(ax,0,value,'C1');ax.axhline(0,color='black',ls='--');ax.set_xlim(-.4,.4);ax.set_xticks([0],['sym_int minus control_int']);ax.set_title(label);ax.grid(alpha=.2)
 fig.suptitle('Paired five-training-seed differences with unadjusted bootstrap95\nPrimary and absolute(O+M) require upper95<0; action E requires mean decrease, with interval shown.')
 save(fig,'paired_primary_and_symmetry')
 fig,axes=plt.subplots(2,3,figsize=(16,9))
 for ax,c in zip(list(axes.flat)[:4],['forward_05','forward_08','yaw_positive','yaw_negative']):
  v=p['tracking_'+c];m=v['relative_change'];ci=v['paired_ratio_bootstrap95']
  if m is not None:point(ax,0,dict(mean=m*100,bootstrap95=None if ci is None else [100*x for x in ci]),'C1')
  else:ax.text(.5,.5,'NA',transform=ax.transAxes)
  ax.axhline(v['bound']*100,color='red',ls='--');ax.axhline(0,color='black',lw=.5);ax.set_title(c+' tracking relative change');ax.set_ylabel('%');ax.set_xticks([0],['sym_int vs control_int'])
 for ax,c in zip(list(axes.flat)[4:],['forward_03','forward_05']):
  for i,group in enumerate(['control_int','sym_int']):point(ax,i,g[f'combined:{group}:{c}']['fall_rate'],'C'+str(i))
  if c=='forward_03':ax.axhline(.01,color='red',ls='--')
  ax.set_title('bounded combined '+c);ax.set_xticks([0,1],['control_int','sym_int']);ax.set_ylabel('initial-trial fall fraction')
 fig.suptitle('Tracking and fall guards: mean limits and separately reported seed uncertainty\ncombined0.3 requires seed95upper<1%;0.5 mean fall fraction must not exceed new control.')
 save(fig,'tracking_and_fall_guards')
 fig,ax=plt.subplots(figsize=(10,6))
 for i,group in enumerate(['control6000','control_int','sym_int']):
  values=p['action_E']['per_seed'][group];point(ax,i,paired_summary(values),'C'+str(i));ax.plot([i]*5,values,'.',alpha=.6)
 for seed in range(5):ax.plot(range(3),[p['action_E']['per_seed'][group][seed] for group in ['control6000','control_int','sym_int']],lw=.7,alpha=.4)
 ax.set_xticks(range(3),['old control6000','control_int','sym_int']);ax.set_ylabel('mean normalized action symmetry error E');ax.set_title('All models evaluated on the SAME102400 fixed raw states\nTraining reflection+halfcycle; deterministic CPU means; five matched training seeds');ax.grid(alpha=.2)
 save(fig,'fixed_state_action_equivariance')
 clock=load('old_clock_stance_distribution.json');a=np.array(clock['old_counts_by_cycle']);fig,axes=plt.subplots(1,2,figsize=(13,5))
 for i,label in enumerate(['left','right']):axes[0].plot(range(50),a[:,i],'.-',label=label);axes[1].bar(np.array([10,11,12])+(i-.5)*.3,[(a[:,i]==n).sum() for n in [10,11,12]],width=.3,label=label)
 axes[0].axhline(11,color='green',ls='--',label='integer:11bothlegs');axes[0].set_xlabel('cycle');axes[0].set_ylabel('stance steps per20');axes[0].legend();axes[1].set_xticks([10,11,12]);axes[1].set_xlabel('stance steps');axes[1].set_ylabel('number of old-clock cycles');axes[1].legend();fig.suptitle('Old floating clock:40% of50cycles have unequal left/right stance counts\nNew integer clock:11stance steps each, double support exactly indices0and10; all1000steps tested.')
 save(fig,'integer_clock_diagnosis')
 inputs=['group_statistics.json','seed_statistics.json','paired_comparisons.json','old_clock_stance_distribution.json']
 (ROOT/'figure_outputs_sha256.json').write_text(json.dumps(dict(source_sha256=sha(__file__),inputs={n:sha(ROOT/n) for n in inputs},outputs=outputs,figure_count=6),indent=2)+'\n');print('ISSUE17_FIGURES_COMPLETE6',flush=True)
if __name__=='__main__':main()
