"""Reconstruct the original control0.5 true-median heading trial, with exact old-data checks."""
import json,os
from pathlib import Path
import numpy as np
from issue16_common import ROOT,REPO,COMMANDS,run,evaluate_args,sha

def main():
 os.chdir(REPO);assert (ROOT/'formal_evaluation_complete').exists()
 selected=json.loads((ROOT/'representative_selection.json').read_text());cohort=json.loads((ROOT/'cohort.json').read_text())
 model=cohort['models'][selected['model']];folder=ROOT/'reconstruction';folder.mkdir(exist_ok=True);out=folder/'source.json'
 args=evaluate_args(model,selected['eval_seed'],out,commands=COMMANDS,capture=selected['env_id'],capture_command='forward_05',mode='original')
 if not out.exists():run(args,folder/'source.log','reconstruct_original_control05_heading_median')
 now=json.loads(out.read_text());old=json.loads(Path(selected['original_job']).read_text())
 assert now['results']==old['results'],'Original source metrics differ'
 newfolder=Path(now['issue16_tracking']['output'])
 A=REPO/'validation_artifacts/stage1_issue15_20261111'
 oldfolder=A/'clean/moments'/f"{selected['model']}_eval{selected['eval_seed']}"
 array_checks=0
 for command in COMMANDS:
  for suffix in ['.npz','_yaw.npz']:
   before=dict(np.load(oldfolder/(command+suffix)));after=dict(np.load(newfolder/(command+suffix)))
   assert set(before)<=set(after)
   for name,a in before.items():
    b=after[name];assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),(command,suffix,name)
    array_checks+=1
 trajectory=newfolder/'forward_05_trajectory.npz';metadata=trajectory.with_suffix('.json')
 with np.load(trajectory) as z:
  assert z['values'].shape==(1000,43) and z['flags'].shape==(1000,4)
  reset_frames=np.flatnonzero(~z['flags'][:,0].astype(bool)).tolist()
  q=z['values'][:,3:7].astype('float64')
  yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
 with np.load(oldfolder/'forward_05_yaw.npz') as z:
  recorded=z['heading_yaw'][:,selected['env_id']].astype('float64')
  initial_yaw=float(z['heading_start'][selected['env_id']])
 assert np.allclose(np.arctan2(np.sin(yaw-recorded),np.cos(yaw-recorded)),0,atol=2e-6,rtol=0)
 data=json.loads(metadata.read_text());assert data['env_id']==selected['env_id'] and data['command']=='forward_05'
 state=dict(selected,heading_start_yaw=initial_yaw,reset_frames=reset_frames,trajectory=str(trajectory),trajectory_meta=str(metadata),trajectory_metadata=str(metadata),trajectory_sha256=sha(trajectory),trajectory_metadata_sha256=sha(metadata),
  all_prior_original_result_rows_exactly_reproduced=True,all_prior_original_recorded_arrays_bitwise_equal=True,compared_arrays=array_checks,
  extra_stand_series_is_new_observation_only=True,formal_denominators_unchanged=True,source_sha256=sha(__file__))
 (ROOT/'reconstruction_state.json').write_text(json.dumps(state,indent=2)+'\n')
 print('ISSUE16_ORIGINAL_RECONSTRUCTION_EXACT',array_checks,flush=True)
if __name__=='__main__':main()
