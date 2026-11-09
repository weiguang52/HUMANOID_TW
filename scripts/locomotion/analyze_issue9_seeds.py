"""Issue #9: five matched training seeds are the statistical units."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seed_variance import paired_seed_summary,planning_summary,exact_indices,paired_t_power
from confidence_stats import wilson
ROOT=Path('/root/gpufree-data/tw_stage1/issue9_20261109')
REF7=Path('validation_artifacts/stage1_issue7_20261109')
REF8=Path('validation_artifacts/stage1_issue8_20261109')
assert (ROOT/'high_resolution_complete').exists()
assert '\nOK\n' in (ROOT/'statistics_tests.log').read_text()
COMMANDS=['forward_03','forward_05','forward_08']
def csvrows(path):return list(csv.DictReader(path.open()))
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
evaluations=[]
for variant,ref,names in [('requested',REF7,['new_bounded_s1','new_bounded_s2']),('applied',REF8,['applied_s1','applied_s2'])]:
    manifest=json.loads((ref/'sha256.json').read_text())
    assert hashlib.sha256((ref/'rates_wilson95.csv').read_bytes()).hexdigest()==manifest['rates_wilson95.csv']
    old=csvrows(ref/'rates_wilson95.csv')
    for seed,name in enumerate(names,1):
        subset=[x for x in old if x['model']==name];assert len(subset)==9
        for x in subset:evaluations.append({'variant':variant,'training_seed':seed,'eval_seed':int(x['eval_seed']),'command':x['command'],'falls':int(x['falls']),'trials':int(x['trials']),'rate':float(x['rate']),'wilson95_low':float(x['wilson95_low']),'wilson95_high':float(x['wilson95_high']),'source':str(ref/'rates_wilson95.csv')})
for seed in [3,4,5]:
    for variant in ['requested','applied']:
        for eval_seed in [1,2,3]:
            path=ROOT/f'{variant}_s{seed}_eval{eval_seed}.json';d=json.loads(path.read_text())
            assert d['last_action_observation']==variant
            assert d['policy_last_action_params']==({} if variant=='requested' else {'action_name':'JointPositionAction'})
            assert d['critic_last_action_params']=={} and d['paired_resets'] and d['deterministic_policy']
            assert d['mass_randomization']=='bounded'
            c=d['perturbation_config'];assert c['mass_operation']=='scale' and c['mass_distribution_params']==[.5,1.5]
            assert c['push_velocity_range']=={'x':[-.3,.3],'y':[-.3,.3]} and c['action_class']=='RandomOneStepDelayAction'
            assert c['friction_static']==[.4,1.4] and c['friction_dynamic']==[.3,1.2]
            assert [x['command'] for x in d['results']]==COMMANDS
            for x in d['results']:
                assert x['num_envs']==1024 and x['seconds']==20 and x['seed']==eval_seed
                assert all(np.isfinite(v) for v in x['metrics'].values())
                ci=wilson(x['failed_trials'],1024)
                evaluations.append({'variant':variant,'training_seed':seed,'eval_seed':eval_seed,'command':x['command'],'falls':x['failed_trials'],'trials':1024,'rate':x['failed_trials']/1024,'wilson95_low':ci[0],'wilson95_high':ci[1],'source':str(path)})
assert len(evaluations)==90 and sum(x['trials'] for x in evaluations)==92160
write_csv(ROOT/'all_evaluation_rates.csv',evaluations)
seed_rows=[];pairs=[];statistics={};planning={}
for command in COMMANDS:
    values={}
    for variant in ['requested','applied']:
        values[variant]=[]
        for seed in [1,2,3,4,5]:
            rows=[x for x in evaluations if x['variant']==variant and x['training_seed']==seed and x['command']==command]
            assert len(rows)==3 and {x['eval_seed'] for x in rows}=={1,2,3}
            k=sum(x['falls'] for x in rows);n=sum(x['trials'] for x in rows);assert n==3072
            rate=k/n;values[variant].append(rate)
            seed_rows.append({'variant':variant,'training_seed':seed,'command':command,'falls':k,'evaluation_trials':n,'fall_rate':rate})
    statistics[command]=paired_seed_summary(values['requested'],values['applied'])
    planning[command]=planning_summary(values['requested'],values['applied'])
    for seed,r,a in zip([1,2,3,4,5],values['requested'],values['applied']):
        pairs.append({'training_seed':seed,'command':command,'requested_rate':r,'applied_rate':a,'applied_minus_requested':a-r,'difference_percentage_points':100*(a-r)})
write_csv(ROOT/'training_seed_rates.csv',seed_rows);write_csv(ROOT/'seed_pair_differences.csv',pairs)
(ROOT/'seed_level_statistics.json').write_text(json.dumps(statistics,indent=2))
(ROOT/'power_planning.json').write_text(json.dumps({'primary_speed':'forward_05','per_speed':planning},indent=2))
groups={}
for variant in ['requested','applied']:
    upper={c:statistics[c][variant]['bootstrap95'][1] for c in COMMANDS[:2]}
    groups[variant]={'low_mid_mean_ci_upper':upper,'both_upper_below1pct':all(v<.01-1e-12 for v in upper.values())}
acceptance={'primary_speed':'forward_05','primary_effect_label':statistics['forward_05']['effect_label'],'per_speed_effects':{c:{'difference95':statistics[c]['paired_applied_minus_requested']['bootstrap95'],'excludes_zero':statistics[c]['difference_ci_excludes_zero'],'effect_label':statistics[c]['effect_label']} for c in COMMANDS},'group_mean_thresholds':groups,'any_group_both_low_mid_upper_below1pct':any(v['both_upper_below1pct'] for v in groups.values()),'defaults_promoted':False,'training_pairs':5,'bootstrap_resamples':3125,'numeric_boundary_tolerance':1e-12,'independent_parent_available':False,'limits':'five pairs; empirical percentile intervals conditional on one shared parent; secondary intervals are not simultaneous familywise95% coverage'}
(ROOT/'acceptance.json').write_text(json.dumps(acceptance,indent=2))
indices=exact_indices(5);arrays={'seed_indices':indices,'commands':np.asarray(COMMANDS)}
for c in COMMANDS:
    r=np.asarray(statistics[c]['requested']['rates']);a=np.asarray(statistics[c]['applied']['rates'])
    arrays[c+'_requested']=r[indices].mean(1);arrays[c+'_applied']=a[indices].mean(1);arrays[c+'_paired_difference']=(a-r)[indices].mean(1)
np.savez_compressed(ROOT/'exact_bootstrap_distributions.npz',**arrays)
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True);colors=plt.cm.tab10(np.arange(5))
fig,axes=plt.subplots(1,3,figsize=(13,4.8))
for ax,c in zip(axes,COMMANDS):
    s=statistics[c]
    for i,color in enumerate(colors):ax.plot([0,1],[100*s['requested']['rates'][i],100*s['applied']['rates'][i]],'o-',color=color,alpha=.8,label=f'seed{i+1}')
    for x,variant in enumerate(['requested','applied']):
        mean=100*s[variant]['mean'];lo,hi=np.asarray(s[variant]['bootstrap95'])*100
        ax.errorbar([x],[mean],yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='D',color='black',capsize=6,lw=2)
    ax.set_xticks([0,1],['requested','applied']);ax.set_ylabel('fall rate (%)');ax.set_title(c+(' (primary)' if c=='forward_05' else ''));ax.grid(alpha=.2)
axes[0].legend(fontsize=7);fig.suptitle('Five matched training seeds; black: mean and seed-bootstrap95% interval');fig.tight_layout();fig.savefig(FIG/'five_seed_paired_rates.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for x,c in enumerate(COMMANDS):
    d=statistics[c]['paired_applied_minus_requested'];mean=100*d['mean'];lo,hi=np.asarray(d['bootstrap95'])*100
    ax.errorbar([x],[mean],yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='D',color='black',capsize=6,lw=2)
    for i,color in enumerate(colors):ax.scatter(x+(i-2)*.055,100*d['rates'][i],color=color,s=35)
ax.axhline(0,color='gray',ls='--');ax.set_xticks(range(3),COMMANDS);ax.set_ylabel('applied minus requested (percentage points)');ax.set_title('Training-pair differences; negative favors applied');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'paired_difference_intervals.png',dpi=160);plt.close(fig)
p=planning['forward_05'];sd=p['paired_difference_sample_sd'];needed=p['paired_training_pairs_needed']
if sd>1e-12:
    maximum=min(500,max(30,int((needed or 100)*1.5)));ns=np.arange(2,maximum+1)
    y=[paired_t_power(int(n),sd) for n in ns]
    fig,ax=plt.subplots(figsize=(9,5));ax.plot(ns,y,label='pilot SD point estimate')
    low,high=p['normal_theory_sd95']
    if low>1e-12:
        ylow=[paired_t_power(int(n),high) for n in ns];yhigh=[paired_t_power(int(n),low) for n in ns];ax.fill_between(ns,ylow,yhigh,alpha=.2,label='normal-theory pilot SD95% range')
    ax.axhline(.8,color='black',ls='--',label='80% target');ax.axvline(5,color='gray',ls=':',label='current5pairs')
    if needed is not None and needed<=maximum:ax.axvline(needed,color='red',ls=':',label=f'planned {needed} pairs')
    ax.set_ylim(0,1.02);ax.set_xlabel('matched training pairs (seeds per variant)');ax.set_ylabel('illustrative two-sided power');ax.set_title('0.5m/s: detecting an absolute3pp difference; normal-difference assumption');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'three_pp_power_planning.png',dpi=160);plt.close(fig)
print('SEED_LEVEL_ANALYSIS_COMPLETE',json.dumps({'primary':acceptance['primary_effect_label'],'primary_difference95':acceptance['per_speed_effects']['forward_05']['difference95'],'primary_planned_pairs':needed}),flush=True)
