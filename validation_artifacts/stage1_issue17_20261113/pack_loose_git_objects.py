"""Lossless bounded-pack Git maintenance; retain every enumerated object, no prune of unreachable data."""
import subprocess,json,os
from pathlib import Path
repo=Path('/root/gpufree-data/projects/HUMANOID_TW');os.chdir(repo);objects=repo/'.git/objects';rows=[]
for d in objects.iterdir():
 if len(d.name)==2 and all(c in '0123456789abcdef' for c in d.name):
  for f in d.iterdir():
   if len(f.name)==38:rows.append((d.name+f.name,f.stat().st_size))
batches=[];current=[];size=0
for oid,n in rows:
 if current and size+n>256*1024*1024:batches.append(current);current=[];size=0
 current.append(oid);size+=n
if current:batches.append(current)
for i,batch in enumerate(batches):
 result=subprocess.check_output(['git','pack-objects','--window=20','--depth=30','.git/objects/pack/pack'],input=('\n'.join(batch)+'\n').encode()).decode().strip()
 subprocess.run(['git','verify-pack',str(objects/'pack'/('pack-'+result+'.idx'))],check=True)
 subprocess.run(['git','prune-packed'],check=True)
 print('VERIFIED_PACK',i+1,len(batches),len(batch),result,flush=True)
print('LOSSLESS_GIT_PACKING_COMPLETE',len(rows),flush=True)
