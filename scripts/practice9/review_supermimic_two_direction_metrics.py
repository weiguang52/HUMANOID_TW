"""Review final native replay pair against unchanged development gates."""
import argparse,json,hashlib,statistics,datetime
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--all-component',action='store_true');p.add_argument('--motion-action',action='store_true');p.add_argument('--motion-center',action='store_true');p.add_argument('--hidden-moment',action='store_true');p.add_argument('--linear-center',action='store_true');p.add_argument('--dense-center',action='store_true');p.add_argument('--moment-reset',action='store_true');p.add_argument('--step-guard',action='store_true');p.add_argument('--h32',action='store_true');p.add_argument('--reference-root',type=Path);a=p.parse_args();r=a.root;out=r/'paired_metric_review.json';assert not out.exists()
integrity=json.loads((r/('fit_integrity.json' if (a.all_component or a.moment_reset or a.motion_action or a.motion_center or a.dense_center or a.linear_center or a.hidden_moment) else 'final_native_integrity.json')).read_text());assert integrity['passed']
plans=[json.loads((r/arm/'fit/plan.json').read_text()) for arm in ['control','candidate']];diff={k for k in set(plans[0])|set(plans[1]) if plans[0].get(k)!=plans[1].get(k)};assert diff==({'shared_hidden_first_moment_reset_enabled'} if a.hidden_moment else {'motion_center_linear_enabled'} if a.linear_center else {'motion_center_dense_enabled'} if a.dense_center else {'motion_action_center_enabled'} if a.motion_center else {'motion_action_response_enabled'} if a.motion_action else {'first_moment_reset_enabled'} if a.moment_reset else {'two_direction_step'} if a.all_component else {'static_step_guard'} if a.step_guard else {'native_h32_mix','native_h32_short_batch','native_h32_long_batch'} if a.h32 else {'native_replay_loss'})
if a.hidden_moment:
 assert a.h32 and a.reference_root and not any([a.linear_center,a.dense_center,a.motion_center,a.motion_action,a.moment_reset,a.all_component,a.step_guard])
 assert plans[0]['shared_hidden_first_moment_reset_enabled'] is False and plans[1]['shared_hidden_first_moment_reset_enabled'] is True
 assert all(x['motion_center_linear_enabled'] is True and x['motion_center_dense_enabled'] is True and x['motion_action_center_enabled'] is True and x['motion_action_response_enabled'] is True and x['static_step_guard'] is False and x['resume_optimizer_steps']==1000 for x in plans)
elif a.linear_center:
 assert a.h32 and a.reference_root and not any([a.dense_center,a.motion_center,a.motion_action,a.moment_reset,a.all_component,a.step_guard])
 assert plans[0]['motion_center_linear_enabled'] is False and plans[1]['motion_center_linear_enabled'] is True
 assert all(x['motion_center_dense_enabled'] is True and x['motion_action_center_enabled'] is True and x['motion_action_response_enabled'] is True and x['static_step_guard'] is False for x in plans)
elif a.dense_center:
 assert a.h32 and a.reference_root and not any([a.motion_center,a.motion_action,a.moment_reset,a.all_component,a.step_guard])
 assert plans[0]['motion_center_dense_enabled'] is False and plans[1]['motion_center_dense_enabled'] is True
 assert all(x['motion_action_center_enabled'] is True and x['motion_action_response_enabled'] is True and x['static_step_guard'] is False for x in plans)
elif a.motion_center:
 assert a.h32 and a.reference_root and not any([a.motion_action,a.moment_reset,a.all_component,a.step_guard])
 assert plans[0]['motion_action_center_enabled'] is False and plans[1]['motion_action_center_enabled'] is True
 assert all(x['motion_action_response_enabled'] is True and x['static_step_guard'] is False for x in plans)
elif a.motion_action:
 assert a.h32 and a.reference_root and not a.all_component and not a.step_guard and not a.moment_reset
 assert plans[0]['motion_action_response_enabled'] is False and plans[1]['motion_action_response_enabled'] is True
 assert plans[0]['static_step_guard'] is False and plans[1]['static_step_guard'] is False
elif a.moment_reset:
 assert a.h32 and a.reference_root and not a.all_component and not a.step_guard
 assert plans[0]['first_moment_reset_enabled'] is False and plans[1]['first_moment_reset_enabled'] is True
 assert plans[0]['static_step_guard'] is True and plans[1]['static_step_guard'] is True
elif a.all_component:
 assert a.h32 and a.reference_root and not a.step_guard and plans[0]['two_direction_step'] is False and plans[1]['two_direction_step'] is True
elif a.step_guard:
 assert a.h32 and a.reference_root and plans[0]['static_step_guard'] is False and plans[1]['static_step_guard'] is True
elif a.h32:
 assert a.reference_root and plans[0]['native_h32_mix'] is False and plans[1]['native_h32_mix'] is True
else:assert plans[0]['native_replay_loss'] is False and plans[1]['native_replay_loss'] is True
hashes={arm:hashlib.sha256((r/'declared'/arm/'fit/world_model_calibrated.pt').read_bytes()).hexdigest() for arm in ['control','candidate']}
metrics=['root_position_error_p95_m','root_orientation_error_p95_rad','root_linear_velocity_rmse_m_s','root_angular_velocity_rmse_rad_s','joint_position_rmse_rad','joint_velocity_rmse_rad_s'];summary={}
kinds=['static','dynamic','expanded']+(['original_static','original_dynamic'] if a.h32 else [])
for kind in kinds:
 basekind=kind.removeprefix('original_');reference=a.reference_root if kind.startswith('original_') else r
 reports={'control':json.loads((reference/'control'/(basekind+'_after.json')).read_text()),'candidate':json.loads((r/'candidate'/(basekind+'_after.json')).read_text())}
 expected_control=hashlib.sha256((reference/'declared/control/fit/world_model_calibrated.pt').read_bytes()).hexdigest()
 assert reports['control']['checkpoint_sha256']==expected_control and reports['candidate']['checkpoint_sha256']==hashes['candidate']
 assert [(x['trace'],x['split']) for x in reports['control']['rows']]==[(x['trace'],x['split']) for x in reports['candidate']['rows']]
 checks=[]
 for horizon in ['1','8','32']:
  values={};nonfinite=[]
  for arm,report in reports.items():
   rows=[x['world_model'][horizon] for x in report['rows'] if x['split']==('fit' if kind=='expanded' else 'validation') and x['world_model'][horizon]['windows']>0]
   assert rows
   if any(x.get('finite_count',0)!=x['windows'] for x in rows):nonfinite.append(arm);continue
   values[arm]=dict(windows=sum(x['windows'] for x in rows),metrics={m:statistics.median(x[m] for x in rows) for m in metrics})
  if nonfinite:checks.append(dict(horizon=int(horizon),metric='finite_rollouts',failed_arms=nonfinite,passed=False));continue
  assert values['control']['windows']==values['candidate']['windows']
  for metric in metrics:
   c=values['control']['metrics'][metric];d=values['candidate']['metrics'][metric];checks.append(dict(horizon=int(horizon),metric=metric,control=c,candidate=d,limit=1.05*c,passed=d<=1.05*c))
 summary[kind]=dict(checks=checks,failed=sum(not x['passed'] for x in checks),scope='Fit diagnostic only, not generalization gate' if kind=='expanded' else 'Original validation <=5% nonregression gate')
coordinates={arm:json.loads((r/arm/'declared_coordinate_audit.json').read_text())['all_within_structure_atol'] for arm in hashes};branch=json.loads((r/'declared/branch_validation.json').read_text())
original_branch=json.loads((r/'declared/reference_branch_validation.json').read_text()) if a.h32 else branch
passed=all(v['failed']==0 for k,v in summary.items() if k!='expanded') and all(coordinates.values()) and branch['passed'] and original_branch['passed']
guard_steps={}
if a.all_component:
 for arm in ['control','candidate']:
  history=json.loads((r/arm/'fit/two_direction_history.json').read_text());assert len(history)==1000
  guard_steps[arm]=dict(attempts=1000,full_steps=sum(x['selected_factor']==1 for x in history),reduced_steps=sum(0<x['selected_factor']<1 for x in history),zero_steps=sum(x['selected_factor']==0 for x in history))
if a.step_guard or a.moment_reset:
 for arm in ['control','candidate']:
  history=json.loads((r/arm/'fit/static_step_guard_history.json').read_text());assert len(history)==1000
  guard_steps[arm]=dict(attempts=1000,full_steps=sum(x['applied_factor']==1 for x in history),reduced_steps=sum(0<x['applied_factor']<1 for x in history),zero_steps=sum(x['applied_factor']==0 for x in history),same_batch_nonincrease=sum(x['applied_nonincrease'] for x in history))
result=dict(reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),guard_step_counts=guard_steps,original_reference_root=str(a.reference_root) if a.reference_root else None,original_branch_passed=original_branch['passed'],single_variable_verified=True,checkpoint_sha256=hashes,trajectory=summary,coordinate=coordinates,branch_passed=branch['passed'],development_gate_passed=passed,policy_acceptance=False,promotion_approved=False,scope='All final1000 artifacts, original six metrics x3 horizons, <=1.05 validation ratios, original held branch thresholds and declared precision coordinate contract. No selection or policy promotion; supplementary root response, independent fixed7 and3seeds still required even if these gates pass.')
out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(development_gate_passed=passed,failed={k:v['failed'] for k,v in summary.items()},coordinates=coordinates,branch_passed=branch['passed'])))
