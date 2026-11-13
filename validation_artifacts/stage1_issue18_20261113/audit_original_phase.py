import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113');manifest=json.loads((ROOT/'original_reuse_manifest.json').read_text());rows=[]
for job in manifest['jobs']:
 folder=Path(job['moments']);p=folder/'forward_05.npz'
 with np.load(p) as z:
  valid=np.unpackbits(z['packed_masks'][:,0,0,:],axis=-1,count=1024,bitorder='little').astype(bool);counts=z['counts'][1,0];assert valid.shape==(1000,1024)
  counter=np.zeros(1024,dtype=np.int64);phase=np.empty((1000,1024),dtype=np.uint8)
  for t in range(1000):phase[t]=counter%20;counter=np.where(valid[t],counter+1,0)
  assert np.array_equal(valid.sum(0),z['counts'][0,0]);no_resets=valid.all(0)
  if no_resets.all():assert np.array_equal(phase,np.broadcast_to((np.arange(1000)%20)[:,None],phase.shape))
 rows.append({'model':job['model'],'eval_seed':job['eval_seed'],'initial_trials':1024,'valid_nonterminal_samples':int(valid.sum()),'initial_prefix_counts_min':int(counts.min()),'full20s_without_any_reset':int(no_resets.sum()),'terminal_steps':int((~valid).sum()),'phase_reconstructed_from_unchanged_valid_equals_not_dones':True,'source_moments_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
result={'passed':True,'jobs':len(rows),'training_updates':0,'no_original_rollouts_rerun':True,'counter_starts_at_zero_requires_new_runtime_preflight_confirmation':True,'results':rows};(ROOT/'original_phase_reuse_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('ORIGINAL_PHASE_REUSE_AUDIT45',len(rows),'NO_RESET_TRIALS',sum(x['full20s_without_any_reset'] for x in rows),'TERMINAL_STEPS',sum(x['terminal_steps'] for x in rows))
