"""Requested budget curves, frozen gate, seed-level bootstrap intervals."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from issue10_statistics import describe,paired_decline,gate
from confidence_stats import wilson
ROOT=Path('/root/gpufree-data/tw_stage1/issue10_20261109')
COMMANDS=['forward_03','forward_05','forward_08']

def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def check_result(d,seed,legacy=False):
 if legacy:
  assert all(k not in d for k in ['last_action_observation','policy_last_action_params','critic_last_action_params'])
  cfg=yaml.load((Path(d['checkpoint']).parent/'params/env.yaml').read_text(),Loader=yaml.BaseLoader)
  assert cfg['observations']['policy']['last_action']['params']=={} and cfg['observations']['critic']['last_action']['params']=={}
 else:
  assert d['last_action_observation']=='requested' and d['policy_last_action_params']=={} and d['critic_last_action_params']=={}
 assert d['paired_resets'] and d['deterministic_policy'] and d['mass_randomization']=='bounded'
 c=d['perturbation_config'];assert c['mass_operation']=='scale' and c['mass_distribution_params']==[.5,1.5]
 assert c['push_velocity_range']=={'x':[-.3,.3],'y':[-.3,.3]} and c['action_class']=='RandomOneStepDelayAction'
 assert c['friction_static']==[.4,1.4] and c['friction_dynamic']==[.3,1.2]
 assert [x['command'] for x in d['results']]==COMMANDS
 for x in d['results']:
  assert x['num_envs']==1024 and x['seconds']==20 and x['seed']==seed
  assert all(np.isfinite(v) for v in x['metrics'].values())

def analyze():
 models=json.loads((ROOT/'input_models.json').read_text());stages=[500,1000,1500]
 if (ROOT/'stepB_evaluation_complete').exists():stages += [3000,4500]
 evaluations=[];seedrows=[];statistics={}
 for stage in stages:
  for seed in range(1,6):
   name=f'requested_s{seed}_u{stage}';m=models[name]
   for es in [1,2,3]:
    p=ROOT/f'{name}_eval{es}.json';d=json.loads(p.read_text());check_result(d,es,legacy=(stage==1500 and seed<3))
    assert hashlib.sha256(Path(d['checkpoint']).read_bytes()).hexdigest()==m['sha256']
    for x in d['results']:
     k=x['failed_trials'];lo,hi=wilson(k,1024)
     evaluations.append({'training_seed':seed,'target_updates':stage,'actual_logged_updates':m['actual_logged_updates'],'eval_seed':es,'command':x['command'],'falls':k,'trials':1024,'rate':k/1024,'within_policy_wilson95_low':lo,'within_policy_wilson95_high':hi,'source':str(p)})
   for c in COMMANDS:
    rr=[x for x in evaluations if x['training_seed']==seed and x['target_updates']==stage and x['command']==c];assert len(rr)==3
    k=sum(x['falls'] for x in rr);seedrows.append({'training_seed':seed,'target_updates':stage,'actual_logged_updates':m['actual_logged_updates'],'command':c,'falls':k,'evaluation_trials':3072,'fall_rate':k/3072})
  statistics[str(stage)]={c:describe([x['fall_rate'] for x in seedrows if x['target_updates']==stage and x['command']==c]) for c in COMMANDS}
 g=gate(statistics['1000']['forward_05']['rates'],statistics['1500']['forward_05']['rates'])
 if (ROOT/'gate_decision.json').exists():assert json.loads((ROOT/'gate_decision.json').read_text())==g
 else:(ROOT/'gate_decision.json').write_text(json.dumps(g,indent=2))
 write_csv(ROOT/'all_evaluation_rates.csv',evaluations);write_csv(ROOT/'training_seed_curves.csv',seedrows)
 (ROOT/'seed_level_statistics.json').write_text(json.dumps(statistics,indent=2))
 changes={}
 for before,after in zip(stages[:-1],stages[1:]):changes[f'{before}_minus_{after}']={c:paired_decline(statistics[str(before)][c]['rates'],statistics[str(after)][c]['rates']) for c in COMMANDS}
 acceptance={'gate':g,'stepB_executed':4500 in stages,'defaults_promoted':False,'training_seeds':5,'bootstrap_resamples':3125,'limits':'conditional shared parent, five seeds; percentile intervals not simultaneous; finite eval error included in seed SD; gate is operational rule, not proof of convergence'}
 if 4500 in stages:
  acceptance['final4500']={c:statistics['4500'][c] for c in COMMANDS[:2]}
  acceptance['low_speed_mean95_upper_below1pct']=statistics['4500']['forward_03']['bootstrap95'][1]<.01-1e-12
  acceptance['mid_speed_decline1500_minus4500']=paired_decline(statistics['1500']['forward_05']['rates'],statistics['4500']['forward_05']['rates'])
 (ROOT/'paired_changes.json').write_text(json.dumps(changes,indent=2));(ROOT/'acceptance.json').write_text(json.dumps(acceptance,indent=2))
 fig,axes=plt.subplots(1,3,figsize=(14,4.8));colors=plt.cm.tab10(np.arange(5))
 for ax,c in zip(axes,COMMANDS):
  for seed,color in zip(range(1,6),colors):
   vals=[next(x['fall_rate'] for x in seedrows if x['training_seed']==seed and x['target_updates']==u and x['command']==c)*100 for u in stages]
   ax.plot(stages,vals,'o-',alpha=.65,color=color,label=f'seed{seed}')
  means=np.array([statistics[str(u)][c]['mean'] for u in stages])*100;sd=np.array([statistics[str(u)][c]['sample_sd'] for u in stages])*100
  ax.errorbar(stages,means,yerr=sd,fmt='kD--',capsize=5,lw=2,label='group mean +/- sampleSD')
  ax.set_title(c+(' (primary)' if c=='forward_05' else ''));ax.set_xlabel('target updates (500/1000 saved at505/1005)');ax.set_ylabel('fall rate (%)');ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/budget_curves.png',dpi=170);plt.close(fig)
 if 4500 in stages:
  late=[1500,3000,4500];fig,axes=plt.subplots(1,3,figsize=(14,4.8))
  for ax,c in zip(axes,COMMANDS):
   for seed,color in zip(range(1,6),colors):
    vals=[next(x['fall_rate'] for x in seedrows if x['training_seed']==seed and x['target_updates']==u and x['command']==c)*100 for u in late]
    ax.plot(late,vals,'o-',alpha=.7,color=color,label=f'seed{seed}')
   mean=np.array([statistics[str(u)][c]['mean'] for u in late])*100;sd=np.array([statistics[str(u)][c]['sample_sd'] for u in late])*100
   ax.errorbar(late,mean,yerr=sd,fmt='kD--',capsize=5,lw=2,label='mean +/- sampleSD')
   if c=='forward_03':ax.axhline(1,color='gray',ls=':',label='1% reference (acceptance uses CI upper)')
   ax.set_title(c+(' (primary)' if c=='forward_05' else ''));ax.set_xlabel('target updates (3000 saved at3006)');ax.set_ylabel('fall rate (%)');ax.grid(alpha=.2)
  axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures/budget_late_curves.png',dpi=170);plt.close(fig)
 print('ANALYSIS',json.dumps(g),flush=True)
if __name__=='__main__':analyze()
