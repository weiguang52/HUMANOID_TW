"""Apply prospective O/N/H conditions conservatively and retain mixed evidence."""
import json
from issue21_common import ROOT,save,sha

def main():
 pairs=json.loads((ROOT/'pair_analysis.json').read_text());perturb=json.loads((ROOT/'perturbation_analysis.json').read_text());complete=(ROOT/'formal_evaluation_complete').exists();evidence=[];oflags=[]
 for x in pairs:
  if x['kind']=='closed':
   for n in ['ang_vel','gravity','commands','joint_pos_obs','joint_vel_obs','last_action','gait_phase','action']:
    v=x['metrics'][n]['step0_or1_linf']
    if v>1e-5:oflags.append({'case':x['case'],'quantity':n,'step':0,'error':v})
  if x['mode']=='air':
   for n in ['joint_pos','joint_vel','applied_torque']:
    v=x['metrics'][n]['step0_or1_linf']
    if v>1e-5:oflags.append({'case':x['case'],'quantity':n,'step':1,'error':v})
 air=[x for x in pairs if x['mode']=='air'];ground=[x for x in pairs if x['mode']=='ground' and x['kind']=='open'];closed=[x for x in pairs if x['kind']=='closed'];airpass=len(air)==20 and all(x['metrics'][n]['max_linf']<=1e-5 for x in air for n in ['joint_pos','joint_vel','applied_torque'])
 probe_pass=bool(closed) and all(x['counterfactual_action_error_max']<=1e-5 for x in closed)
 smallflip=all(any(x['training_seed']==ts and abs(x['epsilon'])<=.05 and x['majority_flip'] and x['comparison_valid']==1024 for x in perturb) for ts in range(1,6))
 large_no_flip=len(perturb)==30 and all(x['flips']==0 and x['censored']==0 and x['comparison_valid']==1024 for x in perturb if abs(x['epsilon'])==.2)
 starts_small=[];contact_large=[];exp=[]
 for x in ground:
  import numpy as np
  z=np.load(x['residual_file']);first=x['first_contact_post_step']
  for env,step in enumerate(first):
   if step is None:continue
   fields=['root_position','root_quaternion','root_linear_velocity','root_angular_velocity','joint_pos','joint_vel','applied_torque','contact'];err=max(float(z[n+'_linf'][step-1,env]) for n in fields);starts_small.append(err<=1e-5);contact_large.append(err>=1e-4)
  exp.extend(v['exponential'] if v is not None else False for v in x['growth_root_angular_velocity'])
 checks={'complete':complete,'O_components':oflags,'air_full_200_step_pass':airpass,'mirrored_input_policy_probe_pass':probe_pass,'small_epsilon_majority_flip_every_seed':smallflip,'epsilon02_no_single_flip_all_seeds':large_no_flip,'ground_first_contact_all_small':bool(starts_small) and all(starts_small),'ground_first_contact_all_large':bool(contact_large) and all(contact_large),'ground_exponential_all_trials':bool(exp) and all(exp),'counts':{'pairs':len(pairs),'air_pairs':len(air),'ground_open_pairs':len(ground),'perturb_cells':len(perturb),'first_contact_small':sum(starts_small),'first_contact_large':sum(contact_large),'growth_exponential':sum(exp),'growth_trials':len(exp)}}
 if not complete:label='pending'
 elif oflags:label='O'
 elif airpass and probe_pass and smallflip and checks['ground_first_contact_all_small'] and checks['ground_exponential_all_trials']:label='N'
 elif airpass and large_no_flip and checks['ground_first_contact_all_large']:label='H'
 else:label='other'
 recommendation={'O':'Repair the specifically identified non-equivariant pipeline component, then rerun only the symmetric-USD issue20 prerequisite in a future issue.','N':'Add explicit yaw-angular-velocity tracking in future training; it directly constrains the observed long-run turning rate.','H':'Add explicit yaw-angular-velocity tracking in future training; document the measured contact asymmetry for later sim-to-real review.','other':'Investigate the earliest reproducible component and mixed evidence in one targeted follow-up before changing training.','pending':'Await all prescribed diagnostic cells before giving a final recommendation.'}[label]
 save('scientific_conclusion.json',{'classification':label,'checks':checks,'recommendation':recommendation,'recommendation_implemented':False,'training_updates':0,'default_promoted':False,'parameter_scan':False,'rules_sha256':sha(ROOT/'classification_rules.json')});print('ISSUE21_CLASSIFICATION',label,checks['counts'],flush=True)
if __name__=='__main__':main()
