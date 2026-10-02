"""Resumable bounded CPU shards; strict retarget gates remain enabled."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
root=Path('/root/gpufree-data/datasets/practice9/tw75_dataset_v1');rows=json.load(open(root/'selection.json'));batches=[rows[i:i+100] for i in range(0,len(rows),100)]
status=root/f'worker_{a.worker}.status';status.write_text('running\n')
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for i,records in enumerate(batches):
 if i%a.workers!=a.worker:continue
 dest=root/'batches'/f'{i:03d}';dest.mkdir(parents=True,exist_ok=True);manifest=dest/'retarget_manifest.json'
 if manifest.exists():
  old=json.load(open(manifest))
  if len(old.get('motions',[]))+len(old.get('failures',[]))==len(records):continue
 inp=dest/'inputs.txt';inp.write_text(''.join(r['source']+'\n' for r in records))
 with open(dest/'retarget.log','w') as log:
  run=subprocess.run([sys.executable,'scripts/practice9/retarget_humanml3d.py','--input',str(inp),'--output-dir',str(dest/'retargeted'),'--continue-on-error'],env=env,stdout=log,stderr=subprocess.STDOUT)
 if not manifest.exists():
  status.write_text(f'failed:batch{i}:exit{run.returncode}\n');raise SystemExit(1)
 outcome=json.load(open(manifest));print(i,len(outcome.get('motions',[])),len(outcome.get('failures',[])),flush=True)
status.write_text('completed\n')
