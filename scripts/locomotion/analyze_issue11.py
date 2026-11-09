"""Full-denominator risk tables, seed effects, corners and matched-stage diagnostics."""
import csv,json,itertools,collections,math
from pathlib import Path
import numpy as np
from issue11_dataset import ROOT,COMMANDS,load_dataset
from risk_analysis_utils import risk,seed_effect,SEED_INDICES

def write_csv(path,rows):
 assert rows,path
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def counts(rows,outcome):return [[sum(int(x[outcome]) for x in rows if x['training_seed']==s),sum(x['training_seed']==s for x in rows)] for s in range(1,6)]

def level_name(value):return str(value)

TRIAL_FACTORS={'static_q':([1,2,3,4],2),'dynamic_q':([1,2,3,4],2),'mass_q':([1,2,3,4],2),'delay_ms':([0,20],0)}
PUSH_FACTORS={**TRIAL_FACTORS,'amplitude_q':([1,2,3,4],1),'direction':(['forward','back','side'],'forward'),'support_code':([0,1,2,3],3),'phase_bin':([0,1,2,3,4],1)}

def tables(rows,unit,factors,outcome):
 table=[];effects=[];full_effects={};cache={}
 for stage in [1500,4500]:
  for command in COMMANDS:
   cohort=[x for x in rows if x['stage']==stage and x['command']==command];allrisk=risk(sum(int(x[outcome]) for x in cohort),len(cohort))
   for factor,(levels,reference) in factors.items():
    ref=[x for x in cohort if x[factor]==reference];ref_counts=counts(ref,outcome)
    for level in levels:
     rr=[x for x in cohort if x[factor]==level];k=sum(int(x[outcome]) for x in rr);r=risk(k,len(rr));c=counts(rr,outcome);cache[(stage,command,factor,level)]=c
     e=seed_effect(c,ref_counts);key=f'{unit}:{stage}:{command}:{factor}:{level}';full_effects[key]=e
     row={'unit':unit,'stage':stage,'command':command,'factor':factor,'level':level_name(level),**r,'relative_risk_vs_cohort':None if r['rate'] is None or not allrisk['rate'] else r['rate']/allrisk['rate'],'reference_level':level_name(reference),'seed_equal_weight_mean':e.get('target_mean'),'seed_sample_sd':e.get('target_sample_sd'),'sparse_n_below30':len(rr)<30}
     table.append(row)
     if level!=reference:
      ci=e.get('difference_bootstrap95');rci=e.get('relative_risk_bootstrap95')
      effects.append({'unit':unit,'stage':stage,'command':command,'factor':factor,'target_level':level_name(level),'reference_level':level_name(reference),'target_n':len(rr),'reference_n':len(ref),'estimable_all5seeds':e['estimable'],'seed_mean_risk_difference':e.get('risk_difference'),'difference_seed_boot95_low':None if ci is None else ci[0],'difference_seed_boot95_high':None if ci is None else ci[1],'seed_mean_relative_risk':e.get('relative_risk'),'rr_seed_boot95_low':None if rci is None else rci[0],'rr_seed_boot95_high':None if rci is None else rci[1],'zero_reference_bootstrap_replicates':e.get('zero_reference_bootstrap_replicates')})
 comparisons=[]
 for command in COMMANDS:
  for factor,(levels,_) in factors.items():
   for level in levels:
    e=seed_effect(cache[(4500,command,factor,level)],cache[(1500,command,factor,level)]);ci=e.get('difference_bootstrap95')
    comparisons.append({'unit':unit,'command':command,'factor':factor,'level':level_name(level),'seed1500_mean':e.get('reference_mean'),'seed4500_mean':e.get('target_mean'),'difference4500_minus1500':e.get('risk_difference'),'difference_boot95_low':None if ci is None else ci[0],'difference_boot95_high':None if ci is None else ci[1],'estimable_all5seeds':e['estimable']})
 return table,effects,full_effects,comparisons

def joint_table(rows,unit,outcome,factors):
 out=[]
 for stage in [1500,4500]:
  for command in COMMANDS:
   cohort=[x for x in rows if x['stage']==stage and x['command']==command];denom=len(cohort);allk=sum(int(x[outcome]) for x in cohort);counter=collections.defaultdict(lambda:[0,0])
   for x in cohort:
    key=tuple(x[f] for f in factors);counter[key][0]+=int(x[outcome]);counter[key][1]+=1
   for key in itertools.product(*(factors[f] for f in factors)):
    k,n=counter[key];r=risk(k,n);out.append({'unit':unit,'stage':stage,'command':command,**dict(zip(factors,key)),**r,'cohort_exposure_fraction':None if not denom else n/denom,'all_failure_fraction':None if not allk else k/allk,'relative_risk_vs_cohort':None if not allk or not n else (k/n)/(allk/denom),'sparse_n_below30':n<30})
 return out

def corner_analysis(trials,pushes):
 result={}
 for stage in [1500,4500]:
  for command in COMMANDS:
   cohort=[x for x in trials if x['stage']==stage and x['command']==command];events=[x for x in pushes if x['stage']==stage and x['command']==command and x['eligible']];total=sum(x['failed'] for x in cohort);entry={'all_trials':len(cohort),'all_failures':total,'original':risk(total,len(cohort))}
   for corner in ['strict','envelope']:
    exposed=[x for x in cohort if x[corner+'_exposed']];remaining=[x for x in cohort if not x[corner+'_exposed']];proximal=sum(x[corner+'_proximal_failure'] for x in cohort);ep=[x for x in events if x[corner+'_corner']];other=[x for x in events if not x[corner+'_corner']]
    seed_rows=[]
    for s in range(1,6):
     ss=[x for x in cohort if x['training_seed']==s];ff=sum(x['failed'] for x in ss);seed_rows.append({'training_seed':s,'trials':len(ss),'failures':ff,'exposed_trials':sum(x[corner+'_exposed'] for x in ss),'proximal_failures':sum(x[corner+'_proximal_failure'] for x in ss)})
    pf=np.array([x['proximal_failures'] for x in seed_rows]);nf=np.array([x['failures'] for x in seed_rows]);bootnum=pf[SEED_INDICES].sum(1);bootden=nf[SEED_INDICES].sum(1);frac=bootnum/bootden if (bootden>0).all() else None
    entry[corner]={'exposed_trials':len(exposed),'trial_coverage':len(exposed)/len(cohort),'cooccurring_failed_trials':sum(x['failed'] for x in exposed),'cooccurring_failure_fraction':sum(x['failed'] for x in exposed)/total if total else None,'proximal_failures':proximal,'proximal_failure_fraction':proximal/total if total else None,'proximal_failure_fraction_seed_boot95':None if frac is None else np.quantile(frac,[.025,.975]).tolist(),'corner_push_risk':risk(sum(x['failure_within1s'] for x in ep),len(ep)),'other_push_risk':risk(sum(x['failure_within1s'] for x in other),len(other)),'remaining_trial_risk':risk(sum(x['failed'] for x in remaining),len(remaining)),'remaining_seed_comparison_to_original':seed_effect(counts(remaining,'failed'),counts(cohort,'failed')),'per_seed':seed_rows,'selection_caveat':'corner exposure is observed before first failure and depends on survival; whole-trial exclusion is descriptive, not counterfactual or new acceptance'}
   result[f'{stage}:{command}']=entry
 primary=result['4500:forward_03']['envelope'];rare=primary['trial_coverage']<=.2;dominant=primary['proximal_failure_fraction'] is not None and primary['proximal_failure_fraction']>=.8
 decision={'primary_corner':'ENVELOPE','rare_coverage_threshold':.2,'dominant_proximal_failure_threshold':.8,'trial_coverage':primary['trial_coverage'],'proximal_failure_fraction':primary['proximal_failure_fraction'],'concentrated_in_few_predefined_corners':bool(rare and dominant),'classification':'few_predefined_extreme_corners_dominate' if rare and dominant else 'broader_failures_relative_to_predefined_corners','not_exhaustive_over_all_possible_corners':True,'acceptance_unchanged':True,'defaults_promoted':False,'next_single_recommendation':'increase training sampling weight for the measured extreme envelope under unchanged evaluation; not implemented' if rare and dominant else 'test a low-speed post-push recovery reward targeting pitch/COM recovery under unchanged evaluation; not implemented'}
 return result,decision

def recoverability(trials,pushes):
 summary={}
 for stage in [1500,4500]:
  cohort=[x for x in trials if x['stage']==stage and x['command']=='forward_03'];groups=collections.defaultdict(list)
  for x in cohort:groups[(x['eval_seed'],x['env_id'],x['base_mass_ratio'],x['foot_static_min'],x['foot_dynamic_min'],x['delay_ms'])].append(x)
  failures=sum(x['failed'] for x in cohort);mixed=sum(sum(x['failed'] for x in ss) for ss in groups.values() if any(not x['failed'] for x in ss));all5=sum(len(ss)==5 and all(x['failed'] for x in ss) for ss in groups.values())
  summary[str(stage)]={'initial_draw_groups':len(groups),'failed_trials':failures,'failures_whose_identical_initial_mass_friction_delay_has_another_seed_success':mixed,'fraction':mixed/failures if failures else None,'all5_training_seeds_fail_same_initial_draw_groups':all5,'scope':'initial physical draw only; actual push streams/COM/phase trajectories may differ, so not proof about an exact failed state'}
 stress=[]
 for stage in [1500,4500]:
  for command in COMMANDS:
   for support in [0,1,2,3]:
    for failed in [0,1]:
     rr=[x for x in pushes if x['eligible'] and x['stage']==stage and x['command']==command and x['support_code']==support and x['failure_within1s']==failed]
     record={'stage':stage,'command':command,'support_code':support,'failure_within1s':failed,'n_pushes':len(rr)}
     for feature in ['capture_distance_to_support_centers_proxy','friction_stopping_distance_proxy','com_height','amplitude']:
      vals=[x[feature] for x in rr if x[feature]>=0];record[feature+'_median']=float(np.median(vals)) if vals else None;record[feature+'_p90']=float(np.quantile(vals,.9)) if vals else None
     stress.append(record)
 summary['limits']='Capture-point proxy uses COM prevelocity+root delta and instantaneous foot centers, not finite support polygons or future foot placement. Stopping distance assumes horizontal constant mu*g and conservative supporting-foot dynamic mu. These are stress descriptors, not actuator-aware feasibility bounds. Observational policy failure cannot establish physical unrecoverability. Matched initial-draw successes refute only baseline randomization alone being universally impossible.'
 return summary,stress

def select_representatives(cases,pushes,trials):
 low=[x for x in cases if x['stage']==4500 and x['command']=='forward_03'];rank=collections.Counter(x['mode'] for x in low);out=[]
 pushmap={(x['model'],x['eval_seed'],x['command'],x['source_push_row']):x for x in pushes}
 for mode,count in sorted(rank.items(),key=lambda x:(-x[1],x[0])):
  cc=[x for x in low if x['mode']==mode]
  if mode!='no_recent_push':
   def same_mode(x):return f"delay{x['delay_ms']}__{x['direction']}__support{x['support_code']}"==mode
   exposure=[x for x in pushes if x['stage']==4500 and x['command']=='forward_03' and x['eligible'] and same_mode(x)]
   if len(exposure)<20:continue
   cc=[x for x in cc if pushmap[(x['model'],x['eval_seed'],x['command'],x['last_push_row'])]['eligible']]
   if not cc:continue
   median=float(np.median([x['amplitude'] for x in exposure]));cc.sort(key=lambda x:(abs(x['last_push_amplitude']-median),x['training_seed'],x['eval_seed'],x['env_id'],x['failure_step']))
  else:cc.sort(key=lambda x:(x['failure_step'],x['training_seed'],x['eval_seed'],x['env_id']));exposure=[]
  chosen=dict(cc[0]);chosen['checkpoint_sha256']=json.loads((ROOT/'input_models.json').read_text())[chosen['model']]['sha256'];chosen['trial_baseline']=next(x for x in trials if x['trial_id']==chosen['trial_id']);chosen['push_features']=None if mode=='no_recent_push' else pushmap[(chosen['model'],chosen['eval_seed'],chosen['command'],chosen['last_push_row'])];chosen['eligible_push_events']=None if mode=='no_recent_push' else sum(x['failure_within1s'] for x in exposure);chosen['case_count_is_not_the_push_window_numerator']=True;chosen.update(mode=mode,mode_failure_count=count,mode_failure_share=count/len(low),eligible_push_denominator=None if mode=='no_recent_push' else len(exposure));out.append(chosen)
  if len(out)==2:break
 assert len(out)==2
 for x in out:x['selection_scope']='coarse_case_mode'
 full=collections.defaultdict(list)
 for x in pushes:
  if x['stage']==4500 and x['command']=='forward_03' and x['eligible']:
   key=tuple(x[f] for f in ['static_q','mass_q','delay_ms','amplitude_q','direction','support_code']);full[key].append(x)
 rankfull=sorted(full.items(),key=lambda kv:(-sum(x['failure_within1s'] for x in kv[1]),-len(kv[1]),tuple(map(str,kv[0]))))
 for key,rr in rankfull[:2]:
  failures=[x for x in rr if x['failure_within1s']];assert failures
  median=float(np.median([x['amplitude'] for x in rr]));failures.sort(key=lambda x:(abs(x['amplitude']-median),x['training_seed'],x['eval_seed'],x['env_id'],x['push_step']));push=failures[0];case=next(x for x in low if x['trial_id']==push['trial_id']);chosen=dict(case)
  chosen['source_coarse_mode']=chosen['mode'];chosen['mode']='fulljoint_'+'_'.join(map(str,key));chosen['joint_stratum']=dict(zip(['static_q','mass_q','delay_ms','amplitude_q','direction','support_code'],key));chosen['selection_scope']='full_joint';chosen['mode_failure_count']=len(failures);chosen['mode_failure_share']=len(failures)/len(low);chosen['eligible_push_denominator']=len(rr);chosen['eligible_push_events']=len(failures);chosen['case_count_is_not_the_push_window_numerator']=False;chosen['push_features']=push;chosen['trial_baseline']=next(x for x in trials if x['trial_id']==push['trial_id']);chosen['checkpoint_sha256']=json.loads((ROOT/'input_models.json').read_text())[chosen['model']]['sha256'];out.append(chosen)
 assert len(out)==4
 for i,x in enumerate(out,1):(ROOT/f'representative_mode{i}.json').write_text(json.dumps(x,indent=2))
 return out

def main():
 assert '\nOK\n' in (ROOT/'statistics_tests.log').read_text();assert json.loads((ROOT/'observer_equivalence_audit.json').read_text())['passed']
 trials,pushes,cases,cuts=load_dataset();eligible=[x for x in pushes if x['eligible']]
 write_csv(ROOT/'all_trial_records.csv',trials);write_csv(ROOT/'all_push_records.csv',pushes)
 policy=[];overall={}
 for stage in [1500,4500]:
  for command in COMMANDS:
   rr=[x for x in trials if x['stage']==stage and x['command']==command];cc=counts(rr,'failed');rates=np.array([k/n for k,n in cc]);boot=rates[SEED_INDICES].mean(1)
   overall[f'{stage}:{command}']={**risk(sum(k for k,n in cc),sum(n for k,n in cc)),'seed_rates':rates.tolist(),'seed_equal_mean':float(rates.mean()),'seed_sample_sd':float(rates.std(ddof=1)),'seed_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
   for seed,(k,n) in enumerate(cc,1):policy.append({'stage':stage,'command':command,'training_seed':seed,'falls':k,'trials':n,'fall_rate':k/n})
 write_csv(ROOT/'policy_rates.csv',policy);(ROOT/'overall_risks.json').write_text(json.dumps(overall,indent=2))
 t,te,tf,tc=tables(trials,'20s_trial',TRIAL_FACTORS,'failed');p,pe,pf,pc=tables(eligible,'1s_push_window',PUSH_FACTORS,'failure_within1s')
 write_csv(ROOT/'trial_stratified_risks.csv',t);write_csv(ROOT/'push_stratified_risks.csv',p);write_csv(ROOT/'factor_effects.csv',te+pe);write_csv(ROOT/'stage_stratum_comparison.csv',tc+pc);(ROOT/'factor_seed_effects.json').write_text(json.dumps({**tf,**pf},indent=2))
 jt=joint_table(trials,'20s_trial','failed',{'static_q':[1,2,3,4],'mass_q':[1,2,3,4],'delay_ms':[0,20]});jp=joint_table(eligible,'1s_push_window','failure_within1s',{'static_q':[1,2,3,4],'mass_q':[1,2,3,4],'delay_ms':[0,20],'amplitude_q':[1,2,3,4],'direction':['forward','back','side'],'support_code':[0,1,2,3]});write_csv(ROOT/'trial_joint_strata.csv',jt);write_csv(ROOT/'push_joint_strata.csv',jp)
 joint_comparison=[]
 for table in [jt,jp]:
  features=['static_q','mass_q','delay_ms']+(['amplitude_q','direction','support_code'] if table is jp else [])
  grouped={}
  for x in table:grouped[(x['command'],tuple(x[f] for f in features),x['stage'])]=x
  for (command,key,stage),a in grouped.items():
   if stage!=1500:continue
   b=grouped[(command,key,4500)]
   joint_comparison.append({'unit':a['unit'],'command':command,**{f:(key[features.index(f)] if f in features else None) for f in ['static_q','mass_q','delay_ms','amplitude_q','direction','support_code']},'n1500':a['n'],'falls1500':a['falls'],'rate1500':a['rate'],'wilson1500_low':a['wilson95_low'],'wilson1500_high':a['wilson95_high'],'n4500':b['n'],'falls4500':b['falls'],'rate4500':b['rate'],'wilson4500_low':b['wilson95_low'],'wilson4500_high':b['wilson95_high'],'difference4500_minus1500':None if a['rate'] is None or b['rate'] is None else b['rate']-a['rate'],'rr4500_vs1500':None if a['rate'] in [None,0] or b['rate'] is None else b['rate']/a['rate'],'both_stages_have_exposure':a['n']>0 and b['n']>0})
 write_csv(ROOT/'stage_joint_comparison.csv',joint_comparison)
 mode=[]
 for stage in [1500,4500]:
  low=[x for x in cases if x['stage']==stage and x['command']=='forward_03'];counter=collections.Counter(x['mode'] for x in low)
  for label,k in sorted(counter.items(),key=lambda x:(-x[1],x[0])):mode.append({'stage':stage,'command':'forward_03','case_mode':label,'failures':k,'all_low_speed_failures':len(low),'case_share':k/len(low),'definition':'case composition only; exposure-denominator risks in separate tables'})
 write_csv(ROOT/'failure_mode_comparison.csv',mode);(ROOT/'failure_case_records.json').write_text(json.dumps(cases,indent=2))
 mechanisms={}
 for stage in [1500,4500]:
  for command in COMMANDS:
   rr=[x for x in cases if x['stage']==stage and x['command']==command];n=len(rr);sg=sum(abs(x['terminal_pitch'])>=abs(x['terminal_roll']) for x in rr);recent=sum(x['recent_push'] for x in rr);both=sum(x['recent_push'] and abs(x['terminal_pitch'])>=abs(x['terminal_roll']) for x in rr)
   mechanisms[f'{stage}:{command}']={'first_failures':n,'termination_counts':dict(collections.Counter(t for x in rr for t in x['termination_terms'])),'sagittal_terminal_count':sg,'sagittal_terminal_fraction':sg/n if n else None,'recent_push_count':recent,'recent_push_fraction':recent/n if n else None,'recent_push_and_sagittal_count':both,'recent_push_and_sagittal_fraction':both/n if n else None,'near_hard_limit_cases':sum(x['terminal_near_hard_limit_count']>0 for x in rr),'near_limit_is_not_proof_of_hard_stop_contact':True}
 (ROOT/'failure_mechanism_summary.json').write_text(json.dumps(mechanisms,indent=2))
 changes=[];all_modes=sorted({x['mode'] for x in cases if x['command']=='forward_03'})
 for label in all_modes:
  row={'case_mode':label}
  for stage in [1500,4500]:
   cc=[x for x in cases if x['stage']==stage and x['command']=='forward_03'];k=sum(x['mode']==label for x in cc);ee=[x for x in eligible if x['stage']==stage and x['command']=='forward_03' and f"delay{x['delay_ms']}__{x['direction']}__support{x['support_code']}"==label]
   r=risk(sum(x['failure_within1s'] for x in ee),len(ee));row.update({f'case_count{stage}':k,f'case_share{stage}':k/len(cc),f'push_n{stage}':r['n'],f'push_events{stage}':r['falls'],f'push_risk{stage}':r['rate'],f'push_wilson_low{stage}':r['wilson95_low'],f'push_wilson_high{stage}':r['wilson95_high']})
  row['case_count_change4500_minus1500']=row['case_count4500']-row['case_count1500'];row['observed_zero4500_not_proven_eliminated']=row['case_count4500']==0;changes.append(row)
 write_csv(ROOT/'failure_mode_changes.csv',changes)
 from issue11_concentration_cv import analyze as concentration_cv
 secondary=concentration_cv(trials,pushes,ROOT)
 corners,decision=corner_analysis(trials,pushes)
 decision['secondary_held_out_subset_concentration']=any(x['secondary_concentration_detected'] for x in secondary.values())
 decision['overall_measured_subset_assessment']='concentrated_in_primary_or_secondary_measured_subset' if decision['concentrated_in_few_predefined_corners'] or decision['secondary_held_out_subset_concentration'] else 'broader_failures_across_measured_subsets'
 if decision['secondary_held_out_subset_concentration']:decision['next_single_recommendation']='increase training sampling weight for the measured high-risk subset under unchanged evaluation; not implemented'
 (ROOT/'corner_diagnostics.json').write_text(json.dumps(corners,indent=2));(ROOT/'concentration_decision.json').write_text(json.dumps(decision,indent=2))
 recovery,stress=recoverability(trials,pushes);(ROOT/'physical_recoverability_limits.json').write_text(json.dumps(recovery,indent=2));write_csv(ROOT/'physical_stress_summary.csv',stress)
 reps=select_representatives(cases,pushes,trials);(ROOT/'representative_selection_summary.json').write_text(json.dumps(reps,indent=2))
 degenerate={f:{'min':min(x[f] for x in trials),'max':max(x[f] for x in trials),'effect_estimable':max(x[f] for x in trials)-min(x[f] for x in trials)>1e-9} for f in ['left_foot_mass_ratio','right_foot_mass_ratio']};(ROOT/'fixed_foot_mass_audit.json').write_text(json.dumps(degenerate,indent=2))
 (ROOT/'risk_analysis_complete').write_text('complete');print('ANALYSIS_COMPLETE',json.dumps({'trials':len(trials),'pushes':len(pushes),'eligible':len(eligible),'failures':len(cases),'decision':decision}),flush=True)
if __name__=='__main__':main()
