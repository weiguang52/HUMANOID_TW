from pathlib import Path
import sys,json,subprocess,datetime
base=Path(__file__).parent;root=base.parent;results=[]
subprocess.run([sys.executable,str(base/'arm_tracking_diagnostic.py')],check=True)
subprocess.run([sys.executable,str(base/'audit_export_binding.py')],check=True)
for suffix,seed in [('',42),('_seed123',123)]:
 s=root/('velocity_balance'+suffix+'_20261029');assert (s/'evaluation_status').read_text().strip()=='completed'
 subprocess.run([sys.executable,str(base/'audit_runtime_pipeline.py'),str(s)],check=True)
 subprocess.run([sys.executable,str(s/'body_jitter_diagnostic.py')],check=True)
 runtime=json.loads((s/'runtime_pipeline_audit.json').read_text());ret=json.loads((s/'retention_audit.json').read_text())
 flat=json.loads((s/'evaluation/flat_three_seed_summary.json').read_text());main=json.loads((s/'evaluation/metrics.json').read_text());ind=json.loads((s/'independent/paired_outcomes.json').read_text());body=json.loads((s/'body_jitter_candidate_matrix.json').read_text());fidelity=json.loads((s/'evaluation/locomotion_fidelity.json').read_text())
 assert runtime['complete'] and runtime['observed']==65 and main['complete'] and ind['complete']
 if flat['all_complete_same_duration']:assert body['complete'] and fidelity['complete']
 groups={model:{cat:sum(v['clean'] for v in main['rows'] if v['model']==model and v['category']==cat) for cat in sorted({v['category'] for v in main['rows']})} for model in ['accepted','candidate']}
 preserve=all(groups['candidate'][cat]>=n for cat,n in groups['accepted'].items())
 accepted_cases={(v['seed'],v['id']):v for v in main['rows'] if v['model']=='accepted'}
 candidate_cases={(v['seed'],v['id']):v for v in main['rows'] if v['model']=='candidate'}
 assert len(accepted_cases)==len(candidate_cases)==42 and set(accepted_cases)==set(candidate_cases)
 lost_main=[{'evaluation_seed':k[0],'motion_id':k[1],'category':v['category'],'accepted_first_episode_seconds':v['first_episode_seconds'],'candidate_first_episode_seconds':candidate_cases[k]['first_episode_seconds']} for k,v in sorted(accepted_cases.items()) if v['clean'] and not candidate_cases[k]['clean']]
 checks={'all65_runtime_pipeline_reconstructed':runtime['all_observed_pass'],'all24_full_same_duration':flat['complete'] and flat['all_complete_same_duration'],'fixed_slip_and_hf_improved':flat.get('joint_goal_pass',False),'validation_group_counts':preserve,'validation_no_lost_cases':not lost_main,'independent_group_counts':ind['all_group_counts_preserved'],'independent_no_lost_cases':ind['no_lost_successes'],'both20s_retained_exact':ret['complete'] and ret['passed']}
 old=root/('anchor_strength_20261028' if seed==42 else 'anchor_strength_seed123_20261028')/'evaluation/flat_three_seed_summary.json'
 first_order=json.loads(old.read_text()).get('summary',{}).get('candidate');candidate=flat.get('summary',{}).get('candidate')
 change={k:100*(candidate[k]/first_order[k]-1) for k in candidate} if candidate and first_order else None
 result={'training_seed':seed,'checks':checks,'numeric_screen_pass':all(checks.values()),'flat':flat,'validation_groups':groups,'validation_lost_successes':lost_main,'independent_groups':ind['groups'],'independent_lost_successes':ind['lost_successes'],'parent_anchor_strength_change_percent':change,'fidelity_all24':fidelity['summary'].get('all24'),'body_jitter':body['summary'],'full24_fidelity_available':bool(body['complete'] and fidelity['complete']),'arm_tracking_diagnostic':json.loads((s/'arm_tracking_diagnostic.json').read_text())['summary'],'policy_promoted':False}
 (s/'evaluation_audit.json').write_text(json.dumps(result,indent=2));results.append(result)
 print('SEED',seed,checks,'quality',flat.get('candidate_change_percent'))
out={'complete':True,'both_training_seeds_numeric_pass':all(v['numeric_screen_pass'] for v in results),'results':results,'policy_promoted':False,'final_acceptance':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remaining':'Review full-body jitter, fidelity, colored front/side robot videos and original HumanML3D skeletons. Never automatic promotion; fixed numeric screen does not prove final acceptance.'}
(base/'two_training_seed_report.json').write_text(json.dumps(out,indent=2))
