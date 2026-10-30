from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,numpy as np,subprocess,datetime
b=Path(__file__).parent;tasks=[];pending=[]
for suffix in ('','_seed123'):
 s=b.parent/f'failure_lookback{suffix}_20261028'
 for group in ('validation','independent','retention'):
  for view in ('front','side'):
   folder=s/'videos'/group/view;file=folder/('jobs.json' if group=='retention' else 'jobs42.json');log=(folder/'run.log').read_text() if (folder/'run.log').exists() else ''
   for index,job in enumerate(json.load(open(file))):
    p=Path(job['output'])
    if 'BATCH_COMPLETED '+str(p) not in log:pending.append(str(p));continue
    tasks.append((s.name,group,view,index,p))
def check(task):
 state,group,view,index,p=task;raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(p.with_suffix('.mp4')),'-frames:v','3','-vf','scale=320:180','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
 a=np.frombuffer(raw,np.uint8).reshape(-1,180,320,3);assert len(a)==3
 frames=[{'frame':i,'mean':float(f.mean()),'std':float(f.std()),'max':int(f.max()),'nonzero_fraction':float((f!=0).mean()),'colored_pixel_fraction':float(((f.max(-1).astype(int)-f.min(-1))>20).mean())} for i,f in enumerate(a)]
 return {'state':state,'group':group,'view':view,'job_index':index,'output':str(p),'blank_start_frames':[r['frame'] for r in frames if r['max']<=2 and r['mean']<.5],'frames':frames}
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,tasks))
bad=[r for r in rows if r['blank_start_frames']]
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'observed':len(rows),'expected':92,'pending':pending,'complete':len(rows)==92 and not pending,'rows':rows,'blank_cases':bad,'scope':'First3 encoded-frame presence only. No physics, quality-gate or video trimming changes. Material readiness and continuous visuals remain separate.'}
(b/'startup_frame_audit.json').write_text(json.dumps(out,indent=2));print('Checked',len(rows),'pending',len(pending),'blank clips',len(bad))
for r in bad:print(r['state'],r['group'],r['view'],'job_index',r['job_index'],'blank_frames',r['blank_start_frames'])
