"""Validate exactly200 uninterrupted physical steps for all required ground-open cells."""
import json,numpy as np
from pathlib import Path
from issue21_common import ROOT,sha,save

def main():
 fresh=json.loads((ROOT/'frozen_open_noreset_sources.json').read_text())
 for p,d in fresh['files'].items():assert sha(p)==d,p
 records=json.loads((ROOT/'cold_start_records.json').read_text());group=[x for x in records if x['mode']=='ground' and x['kind']=='open'];assert len(group)==40;trials=0;falls=0
 for rec in group:
  meta=json.loads(Path(rec['output']).read_text());assert sha(rec['output'])==rec['sha256'];lifecycle=meta['diagnostic_lifecycle'];assert lifecycle['automatic_resets_disabled_after_initialization'] and lifecycle['termination_conditions_unchanged'] and lifecycle['one_continuous_physical_trajectory_per_env'] and lifecycle['all_environment_episode_steps_at_end']==200
  row=meta['records'][0];z=np.load(row['path']);assert sha(row['path'])==row['sha256'] and len(z['root'])==200;trials+=meta['num_envs'];falls+=int(z['terminated'].any(0).sum())
  for name in z.files:assert np.isfinite(z[name]).all(),(rec['name'],name)
 assert trials==340
 save('independent_open_continuity_audit.json',{'passed':True,'cold_processes':40,'individual_trajectories':340,'physical_control_steps_per_trajectory':200,'automatic_resets_disabled_only_for_this_diagnostic':True,'termination_conditions_evaluated_and_unchanged':True,'fall_flags_retained':falls,'all_episode_counters_end_at200':True,'all_raw_finite':True,'all_sources_hash_verified':True});print('ISSUE21_ALL40_OPEN_CONTINUITY_AUDIT_PASS',trials,falls,flush=True)
if __name__=='__main__':main()
