"""Final source/config/data integrity checks for the no-training issue11 study."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue11_20261110');BASE='5fa882398b4edfb7ac0a93dc5607822c62ae8a43'
assert (ROOT/'analysis_and_replay_complete').exists() and (ROOT/'all_representative_replays_complete').exists()
assert '\nOK\n' in (ROOT/'statistics_tests.log').read_text()
assert json.loads((ROOT/'observer_equivalence_audit.json').read_text())['passed']
assert json.loads((ROOT/'observer_equivalence_audit.json').read_text())['rng_complete_fields']
supplement=json.loads((ROOT/'observer_audit_supplement.json').read_text());source=Path('scripts/locomotion/trial_risk_records.py').read_text();assert hashlib.sha256(source[source.index('class TrialRiskRecorder'):].encode()).hexdigest()==supplement['after']==supplement['formal_recorder_class_source_sha256_before']
raw=json.loads((ROOT/'raw_record_integrity_audit.json').read_text());assert raw['passed'] and raw['all_trials']==61440 and len(raw['command_records'])==60
old=json.loads(Path('validation_artifacts/stage1_issue9_20261109/training_source_identity.json').read_text())['sha256'];sources={}
for name in old:
 if name=='scripts/locomotion/eval_cmd.py':continue
 p=Path(name);h=hashlib.sha256(p.read_bytes()).hexdigest();b=hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+name])).hexdigest();assert h==b,(name,h,b);sources[name]=h
models=json.loads((ROOT/'input_models.json').read_text());assert len(models)==10
for e in models.values():assert hashlib.sha256(Path(e['checkpoint']).read_bytes()).hexdigest()==e['sha256']
frozen=json.loads((ROOT/'frozen_plan_sha256.json').read_text());assert hashlib.sha256((ROOT/'frozen_plan.md').read_bytes()).hexdigest()==frozen['frozen_plan.md']
secondary=json.loads((ROOT/'frozen_secondary_concentration_sha256.json').read_text());assert hashlib.sha256((ROOT/'frozen_secondary_concentration.md').read_bytes()).hexdigest()==secondary['sha256']
comparison=json.loads((ROOT/'baseline_count_comparison.json').read_text());assert len(comparison)==30
same=all(x['equal'] for vv in comparison.values() for x in vv)
# Any count difference must be reported, never silently replaced with old outcomes.
config={'passed':True,'baseline':BASE,'no_training_launched':True,'training_and_config_sources_unchanged':sources,'observer_only_modified_files':['scripts/locomotion/eval_cmd.py','scripts/locomotion/failure_traces.py'],'visualization_only_modified_file':'scripts/locomotion/replay_failure.py (static first-frame render warmup; no physics stepping)','input_checkpoint_hashes_verified':10,'all_60_command_counts_equal_issue10':same,'first_failure_count':raw['all_first_failures'],'raw_world_body_push_and_support_checks_passed':True,'observer_final_states_rng_and_traces_bitwise_equal_smoke':all(x['bitwise_equal'] for c in json.loads((ROOT/'observer_equivalence_audit.json').read_text())['checks'].values() for x in c.values() if isinstance(x,dict) and 'bitwise_equal' in x)}
(ROOT/'delivery_integrity_audit.json').write_text(json.dumps(config,indent=2));print('DELIVERY_INTEGRITY_PASS','baseline_counts_equal',same,'trials',raw['all_trials'])
