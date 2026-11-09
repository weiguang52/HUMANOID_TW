"""Validate raw pre-reset trajectories against termination results."""
from pathlib import Path
import json,hashlib
import numpy as np
import scipy
ROOT=Path('/root/gpufree-data/tw_stage1/issue9_20261109')
total=0;full=0;minimum=51;per_file={}
for path in sorted((ROOT/'traces').glob('*/*.json')):
    meta=json.loads(path.read_text());z=np.load(path.with_suffix('.npz'));values=z['values'];valid=z['valid'];f=meta['fields']
    assert values.shape[0]==len(meta['records'])==meta['failed_trials']==int(z['failed'].sum())
    assert np.isfinite(values).all()
    for i,record in enumerate(meta['records']):
        a=values[i,valid[i]];count=len(a);assert count==record['trace_frames']
        def field(name):return a[:,slice(*f[name])]
        assert np.allclose(np.linalg.norm(field('root_pose_local')[:,3:],axis=1),1.,atol=1e-5)
        if 'base_height' in record['termination_terms']:assert field('base_height')[-1,0]<.080001
        assert np.max(np.abs(field('last_push_delta_world')))<=.30001
        assert np.isin(field('contact'),[0.,1.]).all()
        assert np.isin(field('delay20ms'),[0.,1.]).all()
        assert int(field('delay20ms')[-1,0])*20==record['delay_ms']==int(z['initial_delay20ms'][record['env_id']])*20
        assert np.allclose(field('root_pose_local')[:,2],field('base_height')[:,0])
        total+=1;full+=count==51;minimum=min(minimum,count)
    per_file[str(path.relative_to(ROOT))]={'first_failures':len(meta['records']),'trial_environments':meta['num_envs']}
assert len(per_file)==54 and total==json.loads((ROOT/'failure_mode_summary.json').read_text())['total_first_failures']
models=json.loads((ROOT/'input_models.json').read_text())
assert all(hashlib.sha256(Path(v['checkpoint']).read_bytes()).hexdigest()==v['sha256'] for v in models.values())
report={'passed':True,'formal_trace_files':54,'first_failure_windows':total,'full_1s_windows':full,'minimum_available_frames':minimum,'checkpoint_hashes_unchanged':True,'terminal_pre_reset_heights_validated':True,'actual_push_bounds_and_delay_labels_validated':True,'scipy_version':scipy.__version__,'files':per_file}
(ROOT/'raw_trace_integrity_audit.json').write_text(json.dumps(report,indent=2))
print('PASS',total,'pre-reset windows;',full,'full1s; minframes',minimum,flush=True)
