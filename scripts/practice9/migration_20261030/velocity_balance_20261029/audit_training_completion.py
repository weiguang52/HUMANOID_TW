from pathlib import Path
import json,hashlib,re,datetime
b=Path(__file__).parent;rows=[]
for s in (b,b.parent/'velocity_balance_seed123_20261029'):
 assert (s/'actual_formal/status').read_text().strip()=='trained'
 log=(s/'actual_formal/train.log').read_text();iters=[int(x) for x in re.findall(r'Learning iteration (\d+)/28495',log)];assert iters==list(range(27995,28495)),(s.name,len(iters))
 assert 'Restored scheduler learning rate' in log and "'scheduler_after': 2.2500000000000005e-05" in log
 for file in ('code_sha256.txt','input_sha256.txt'):
  for line in (s/'actual_formal'/file).read_text().splitlines():
   h,p=line.split(maxsplit=1);assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 cp=Path((s/'actual_formal/final_checkpoint').read_text().strip());assert cp.name=='model_28494.pt';a=json.load(open(s/'formal_audit.json'));assert a['passed'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==a['checkpoint_sha256'];rows.append({'state':s.name,'iterations':len(iters),'checkpoint':str(cp),'sha256':a['checkpoint_sha256']})
for p,h in json.load(open(b/'formal_source_freeze.json'))['files'].items():assert hashlib.sha256((Path('/root/gpufree-data/projects/HUMANOID_TW')/p).read_bytes()).hexdigest()==h,p
(b/'training_completion_audit.json').write_text(json.dumps({'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows},indent=2));print('TRAINING_COMPLETION_PASS')
