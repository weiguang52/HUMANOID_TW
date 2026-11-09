"""Complete RNG-state equivalence recheck; formal observer class is unchanged."""
import json
from run_issue11 import ROOT,run,evaluate
models=json.loads((ROOT/'input_models.json').read_text());folder=ROOT/'smoke_complete_rng';folder.mkdir(exist_ok=True)
for kind in ['baseline','expanded']:
 out=folder/f'{kind}.json'
 if not out.exists():
  args=evaluate(models['requested_s1_u4500']['checkpoint'],1,out,64)+['--failure_traces',str(folder/f'{kind}_traces'),'--observation_audit',str(folder/f'{kind}_state')]
  if kind=='expanded':args+=['--trial_risk_records']
  run(args,ROOT/f'smoke_complete_{kind}.log')
run(['python','scripts/locomotion/audit_issue11_smoke.py'],ROOT/'smoke_complete_audit.log')
assert json.loads((ROOT/'observer_equivalence_audit.json').read_text())['rng_complete_fields']
print('COMPLETE_RNG_EQUIVALENCE_PASS',flush=True)
