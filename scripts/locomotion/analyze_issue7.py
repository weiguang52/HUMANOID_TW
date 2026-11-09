"""Issue #7 conditional rate intervals and descriptive first-failure modes."""
import csv,json,math,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from confidence_stats import wilson
ROOT=Path('/root/gpufree-data/tw_stage1/issue7_20261109')
assert (ROOT/'high_resolution_complete').exists()
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
rates=[];events=[];delay_trials=[];exposures=collections.defaultdict(lambda:collections.Counter())
for job in sorted(ROOT.glob('*_eval[123].json')):
    data=json.loads(job.read_text());model=job.stem.rsplit('_eval',1)[0]
    assert data['paired_resets'] and data['deterministic_policy'] and data['mass_randomization']=='bounded'
    for result in data['results']:
        command=result['command'];n=result['num_envs'];k=result['failed_trials']
        assert n==1024 and result['seconds']==20
        ci=wilson(k,n);rates.append({'model':model,'eval_seed':result['seed'],'command':command,'falls':k,'trials':n,'rate':k/n,'wilson95_low':ci[0],'wilson95_high':ci[1]})
        path=ROOT/'traces'/job.stem/(command+'.json')
        meta=json.loads(path.read_text());npz=path.with_suffix('.npz');z=np.load(npz)
        values=z['values'];valid=z['valid'];failed=z['failed']
        assert len(meta['records'])==k==int(z['failed'].sum())
        assert len(set(x['env_id'] for x in meta['records']))==k
        assert values.shape[0]==k and np.isfinite(values).all()
        fields=meta['fields']
        def get(trace,key):return trace[:,slice(*fields[key])]
        for delayed in [0,1]:
            group=z['initial_delay20ms'].astype(bool)==bool(delayed)
            total=int(group.sum());count=int(z['failed'][group].sum());c=wilson(count,total)
            delay_trials.append({'model':model,'eval_seed':result['seed'],'command':command,'delay_ms':20*delayed,'falls':count,'trials':total,'rate':count/total,'wilson95_low':c[0],'wilson95_high':c[1]})
        for group,counts in meta['at_risk_step_counts'].items():
            for bin_,count in enumerate(counts):exposures[group][bin_]+=count
        for i,record in enumerate(meta['records']):
            trace=values[i][valid[i]]
            assert len(trace)==record['trace_frames'] and len(trace)<=51
            roll=float(get(trace,'roll')[-1,0]);pitch=float(get(trace,'pitch')[-1,0])
            age=float(get(trace,'seconds_since_push')[-1,0]);push=get(trace,'last_push_delta_body')[-1,:2]
            mag=float(np.linalg.norm(get(trace,'last_push_delta_world')[-1,:2]))
            recent=0<=age<=1
            tilt='upright' if max(abs(roll),abs(pitch))<math.radians(5) else ('lateral' if abs(roll)>=abs(pitch) else 'sagittal')
            mode=('recent_push' if recent else 'not_recent_push')+'__'+tilt
            if age<0:direction='no_push';magnitude='no_push'
            else:
                direction=('forward' if push[0]>=0 else 'backward') if abs(push[0])>=abs(push[1]) else ('left' if push[1]>=0 else 'right')
                magnitude='lt0.15' if mag<.15 else ('0.15to0.30' if mag<.30 else 'ge0.30')
            contacts=get(trace,'contact')[-1].astype(int)
            at_push=get(trace,'contact_at_push')[-1].astype(int)
            names={0:'flight',1:'left_only',2:'right_only',3:'double'}
            phase=names[int(contacts[0]+2*contacts[1])]
            pushphase='no_push' if age<0 else names[int(at_push[0]+2*at_push[1])]
            bounds=np.asarray(meta['hard_joint_limits']);positions=get(trace,'joint_pos')
            touching=(positions<=bounds[:,0]+.001)|(positions>=bounds[:,1]-.001)
            outside=(positions<bounds[:,0]-1e-5)|(positions>bounds[:,1]+1e-5)
            hard=get(trace,'joint_near_hard_limit').astype(bool);soft=get(trace,'joint_near_soft_limit').astype(bool)
            event={'model':model,'eval_seed':result['seed'],'command':command,'env_id':record['env_id'],
                'failure_second':record['failure_second'],'termination_terms':';'.join(record['termination_terms']),
                'mode':mode,'tilt_axis':tilt,'terminal_roll_deg':math.degrees(roll),'terminal_pitch_deg':math.degrees(pitch),
                'terminal_height':float(get(trace,'base_height')[-1,0]),'delay_ms':record['delay_ms'],
                'push_age_s':age,'recent_push':recent,'push_direction_body':direction,'push_magnitude_mps':mag,'push_magnitude_bin':magnitude,
                'contact_at_failure':phase,'contact_at_push':pushphase,'hard_limit_any_last1s':bool(hard.any()),'soft_limit_any_last1s':bool(soft.any()),
                'hard_limit_touch_proxy_last1s':bool(touching.any()),'outside_hard_limits_last1s':bool(outside.any()),
                'hard_limit_joints':';'.join(n for n,b in zip(meta['joint_names'],hard.any(0)) if b),
                'trace_frames':len(trace),'source_metadata':str(path),'source_npz':str(npz),'source_metadata_relative':str(path.relative_to(ROOT)),'source_npz_relative':str(npz.relative_to(ROOT)),'record_index':i}
            events.append(event)
assert len(rates)==36
write_csv(ROOT/'rates_wilson95.csv',rates);write_csv(ROOT/'delay_trial_rates.csv',delay_trials);write_csv(ROOT/'failure_events.csv',events)
pooled=[]
for model in sorted(set(x['model'] for x in rates)):
    for command in ['forward_03','forward_05','forward_08']:
        rs=[x for x in rates if x['model']==model and x['command']==command];assert len(rs)==3
        n=sum(x['trials'] for x in rs);k=sum(x['falls'] for x in rs);ci=wilson(k,n)
        pooled.append({'model':model,'command':command,'falls':k,'trials':n,'rate':k/n,'wilson95_low':ci[0],'wilson95_high':ci[1]})
write_csv(ROOT/'pooled_rates_wilson95.csv',pooled)
groups={}
for field in ['mode','termination_terms','push_direction_body','push_magnitude_bin','delay_ms','contact_at_push','contact_at_failure','hard_limit_any_last1s','soft_limit_any_last1s','hard_limit_touch_proxy_last1s','outside_hard_limits_last1s']:
    counts=collections.Counter(str(x[field]) for x in events)
    groups[field]={k:{'failures':v,'fraction_of_failures':v/len(events)} for k,v in counts.most_common()}
models=sorted(set(x['model'] for x in events));modes=collections.Counter(x['mode'] for x in events)
top=modes.most_common(2)
dominant=top[0][0] if top[0][1]/len(events)>=.5 and all(collections.Counter(x['mode'] for x in events if x['model']==m).most_common(1)[0][0]==top[0][0] for m in models) else None
summary={'total_first_failures':len(events),'total_trials':sum(x['trials'] for x in rates),'groups':groups,'dominant_mode':dominant,'top_two_modes':top,'rule':'at least50% overall and ranked first in all4 policies; mode is descriptive, not causal','at_risk_step_counts':{k:dict(v) for k,v in exposures.items()},'step_dt':.02}
(ROOT/'failure_mode_summary.json').write_text(json.dumps(summary,indent=2))
strata=[]
for model in models:
    for command in ['forward_03','forward_05','forward_08']:
        subset=[x for x in events if x['model']==model and x['command']==command]
        counts=collections.Counter(x['mode'] for x in subset)
        for mode,count in counts.items():strata.append({'model':model,'command':command,'mode':mode,'failures':count,'all_failures_in_stratum':len(subset),'fraction':count/len(subset)})
write_csv(ROOT/'mode_by_policy_speed.csv',strata)
plt.figure(figsize=(11,4.8))
labels=list(groups['mode']);values=[groups['mode'][k]['fraction_of_failures']*100 for k in labels]
plt.bar(range(len(labels)),values,color='#4677a9');plt.xticks(range(len(labels)),[x.replace('__','\n').replace('_',' ') for x in labels],fontsize=8);plt.ylabel('% of first failures');plt.title('Descriptive terminal modes; no causal attribution');plt.tight_layout();plt.savefig(FIG/'failure_mode_shares.png',dpi=160);plt.close()
representatives=[]
for rank,(mode,count) in enumerate(top,1):
    eligible=[x for x in events if x['mode']==mode and x['trace_frames']==51]
    assert eligible
    median=float(np.median([x['failure_second'] for x in eligible]))
    event=min(eligible,key=lambda x:(abs(x['failure_second']-median),x['source_metadata'],x['env_id']))
    selection={'rank':rank,'mode':mode,'count':count,'fraction_of_failures':count/len(events),'selection_rule':'full1s window, closest to median failure time, path/env-id tie break',**event}
    selection_path=ROOT/f'representative_mode{rank}.json';selection_path.write_text(json.dumps(selection,indent=2));representatives.append(str(selection_path))
    meta=json.loads(Path(event['source_metadata']).read_text());z=np.load(event['source_npz']);trace=z['values'][event['record_index']];fields=meta['fields']
    def col(key):return trace[:,slice(*fields[key])]
    t=np.arange(-50,1)*.02
    fig,axes=plt.subplots(4,1,figsize=(10,9),sharex=True)
    axes[0].plot(t,col('base_height')[:,0],label='base height');axes[0].axhline(.08,color='r',ls='--',label='height termination threshold');axes[0].set_ylabel('m')
    axes[1].plot(t,np.degrees(col('roll')[:,0]),label='roll');axes[1].plot(t,np.degrees(col('pitch')[:,0]),label='pitch');axes[1].set_ylabel('deg')
    axes[2].step(t,col('contact')[:,0],where='post',label='left contact');axes[2].step(t,col('contact')[:,1]+1.3,where='post',label='right contact (+1.3)');axes[2].set_ylabel('contact')
    axes[3].plot(t,col('joint_near_hard_limit').sum(-1),label='hard-limit-near joints');axes[3].plot(t,col('joint_near_soft_limit').sum(-1),label='soft-limit-near joints');axes[3].set_ylabel('count');axes[3].set_xlabel('seconds relative to first termination')
    age=event['push_age_s']
    for ax in axes:
        if 0<=age<=1:ax.axvline(-age,color='purple',ls=':',label='latest push')
        ax.axvline(0,color='black',lw=.7);ax.legend(loc='best',fontsize=7);ax.grid(alpha=.2)
    fig.suptitle(f"Mode{rank}: {mode}\n{event['model']} eval{event['eval_seed']} {event['command']} env{event['env_id']}; {event['termination_terms']}",fontsize=10)
    fig.tight_layout();fig.savefig(FIG/f'mode{rank}_representative_trace.png',dpi=150);plt.close(fig)
(ROOT/'representatives.json').write_text(json.dumps(representatives,indent=2))
print('ANALYSIS_COMPLETE',json.dumps({'failures':len(events),'dominant':dominant,'top_two':top}),flush=True)
