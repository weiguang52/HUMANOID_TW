"""Versioned offline projection; retains all source clips and failures."""
import pathlib,json,concurrent.futures,traceback
from contact_ik import process
S=pathlib.Path('/root/gpufree-data/datasets/practice9/tw195_v1')
def job(item):
 split,r=item;p=S/'corrected'/split/(r['id']+'.npz')
 try:
  if p.exists() and p.with_suffix('.audit.json').exists():a=json.load(open(p.with_suffix('.audit.json')))
  else:a=process(r,p)
  return dict(split=split,**a)
 except Exception as e:return dict(split=split,id=r['id'],error=str(e),trace=traceback.format_exc())
if __name__=='__main__':
 tasks=[]
 for split,file in [('train','train.json'),('validation','validation/manifest.json')]:
  m=json.load(open('/root/gpufree-data/datasets/practice9/tw154_v1/'+file));tasks.extend((split,r) for r in m['motions'])
 results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
  for a in pool.map(job,tasks,chunksize=1):
   results.append(a)
   if len(results)%20==0:print('processed',len(results),'/',len(tasks),flush=True)
   (S/'build_audit.json').write_text(json.dumps(results,indent=2))
 print('completed',len(results),flush=True)
