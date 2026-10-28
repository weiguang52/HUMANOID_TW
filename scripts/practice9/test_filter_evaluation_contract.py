import sys,json,copy,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from filter_evaluation_contract import validate_jobs,validate_checkpoint_jobs
r=Path('/root/gpufree-data/datasets/practice9')
for suffix in ['', '_seed123']:
 s=r/('filter_aware_training'+suffix+'_20261026');ckpt=Path((s/'actual_smoke/final_checkpoint').read_text().strip())
 jobs=json.loads((r/('mild_leg_filter'+suffix+'_eval_20261026')/'candidate/jobs42.json').read_text())
 result=validate_checkpoint_jobs(ckpt,jobs);assert result['matched_batch_jobs']==14
 leg=result['leg'];ankle=result['ankle']
 for field,value in [('candidate_leg_filter_alpha',.7),('candidate_leg_filter_mode','teacher_residual'),('leg_filter_alpha',.6),('leg_filter_scope','legs'),('leg_filter_mode','reference_residual'),('controller','reference_pd'),('candidate_hip_filter_alpha',.5)]:
  bad=copy.deepcopy(jobs);bad[0][field]=value
  try:validate_jobs(leg,ankle,bad)
  except ValueError:pass
  else:raise AssertionError('Invalid job accepted: '+field)
 bad=copy.deepcopy(jobs);del bad[0]['reference_speed_teacher_export']
 try:validate_jobs(leg,ankle,bad)
 except ValueError:pass
 else:raise AssertionError('Missing transform builder accepted')
 try:validate_checkpoint_jobs(ckpt,[])
 except ValueError:pass
 else:raise AssertionError('Unfiltered single replay accepted')
 old=Path((r/('mean_curvature'+suffix+'_20261026')/'actual_formal/final_checkpoint').read_text().strip())
 assert validate_checkpoint_jobs(old,[]) is None
print('Both real saved training contracts accept correct14 jobs, reject mismatches and leave old checkpoints unchanged')
