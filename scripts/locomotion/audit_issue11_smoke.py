"""Independent observer-equivalence test against existing FailureRecorder."""
import json,argparse
from pathlib import Path
import numpy as np
ROOT=Path('/root/gpufree-data/tw_stage1/issue11_20261110')
parser=argparse.ArgumentParser();parser.add_argument('--directory',default='smoke_complete_rng');args=parser.parse_args();SMOKE=ROOT/args.directory
checks={}
for label in ['forward_03','forward_05']:
 a=np.load(SMOKE/'baseline_state'/f'{label}.npz');b=np.load(SMOKE/'expanded_state'/f'{label}.npz');record={}
 assert {'numpy_rng_aux','numpy_rng_algorithm','python_rng_version','python_rng_gauss_present','python_rng_gauss_value'}<=set(a.files)
 for k in a.files:
  exact=np.array_equal(a[k],b[k]);error=float(np.max(np.abs(a[k].astype(float)-b[k].astype(float))))
  if 'rng' in k:assert exact,(label,k,error)
  else:assert np.allclose(a[k],b[k],atol=1e-6,rtol=1e-6),(label,k,error)
  record[k]={'bitwise_equal':exact,'max_abs_error':error}
 a=np.load(SMOKE/'baseline_traces'/f'{label}.npz');b=np.load(SMOKE/'expanded_traces'/f'{label}.npz')
 for k in a.files:assert np.array_equal(a[k],b[k]),(label,k)
 t=np.load(SMOKE/'expanded_traces'/f'{label}_trials.npz');assert len(t['env_ids'])==64 and np.array_equal(t['failed'],b['failed']);assert np.isfinite(t['pushes']).all()
 meta=json.loads((SMOKE/'expanded_traces'/f'{label}_trials.json').read_text());assert meta['torch_rng_invariance_checks']>=2
 record['all_trials']=64;record['all_pushes']=len(t['pushes']);record['failure_trace_equal']=True;checks[label]=record
(ROOT/'observer_equivalence_audit.json').write_text(json.dumps({'passed':True,'checks':checks,'tolerance':1e-6,'rng_complete_fields':True,'source_directory':str(SMOKE)},indent=2));print('OBSERVER_EQUIVALENCE_PASS',json.dumps(checks),flush=True)
