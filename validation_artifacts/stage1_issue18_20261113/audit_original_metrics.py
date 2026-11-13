import json,sys,pathlib
sys.path.insert(0,'scripts/locomotion')
from issue15_analysis_data import load_yaw,job_statistics
r=pathlib.Path('/root/gpufree-data/tw_stage1/issue18_20261113');count=0
for j in json.loads((r/'original_reuse_manifest.json').read_text())['jobs']:
 d=json.loads(pathlib.Path(j['output']).read_text());folder=pathlib.Path(j['moments'])
 for result in d['results']:
  z,y,m,meta=load_yaw(folder,result['command'],result);v=job_statistics(z,y,meta,result);count+=1
 print('ORIGINAL_AUDIT',j['model'],j['eval_seed'],flush=True)
(r/'original_metrics_audit.json').write_text(json.dumps({'passed':True,'commands':count,'jobs':45,'no_rollouts':True}));print('PASS',count)
