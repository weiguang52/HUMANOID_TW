"""Bind every reused issue19 scientific byte to an existing committed archive."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def digest(p):
 data=Path(p).read_bytes()
 return hashlib.sha256(data).hexdigest(),hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),len(data)
def main():
 rows=subprocess.check_output(['git','ls-tree','-rz','HEAD','--','validation_artifacts']).split(b'\0')
 tracked={}
 for row in rows:
  if not row:continue
  head,path=row.split(b'\t',1);mode,kind,oid=head.decode().split();assert kind=='blob'
  tracked[path.decode()]=oid
 refs=json.loads((ROOT/'reference_reuse_manifest.json').read_text())['jobs'];sources={}
 for job in refs:
  for p,h in {job['output']:job['output_sha256'],**job['source_files_sha256']}.items():
   assert p not in sources or sources[p]==h;sources[p]=h
 portable={};archives={}
 for name in ['issue17_20261113','issue18_20261113']:
  archive=REPO/'validation_artifacts'/('stage1_'+name)
  archives[name]=(archive,json.loads((archive/'lossless_chunk_manifest.json').read_text()))
 for path,expected in sorted(sources.items()):
  p=Path(path);assert digest(p)[0]==expected
  name=next(n for n in archives if p.is_relative_to(ROOT.with_name(n)))
  archive,chunks=archives[name];rel=str(p.relative_to(ROOT.with_name(name)))
  target=archive/rel
  if target.is_file():
   sha,oid,size=digest(target);rp=str(target.relative_to(REPO))
   assert sha==expected and tracked[rp]==oid
   portable[path]=dict(sha256=sha,bytes=size,committed_relative_path=rp,git_blob=oid)
  else:
   c=chunks[rel];assert c['original_sha256']==expected
   combined=hashlib.sha256();size=0;parts=[]
   for part in c['parts']:
    target=archive/part['path'];sha,oid,n=digest(target);rp=str(target.relative_to(REPO))
    assert sha==part['sha256'] and tracked[rp]==oid
    combined.update(target.read_bytes());size+=n;parts.append(dict(committed_relative_path=rp,sha256=sha,git_blob=oid))
   assert size==c['original_bytes'] and combined.hexdigest()==expected
   portable[path]=dict(sha256=expected,bytes=size,lossless_committed_parts=parts)
 models={}
 for r in json.loads((ROOT.with_name('issue17_20261113')/'training_records.json').read_text()):
  wanted=REPO/'validation_artifacts/stage1_issue17_20261113/models'/(r['group']+'_s'+str(r['training_seed'])+'.pt')
  sha,oid,n=digest(wanted);rp=str(wanted.relative_to(REPO))
  assert sha==r['sha256'] and tracked[rp]==oid and digest(r['checkpoint'])[0]==sha
  models[r['group']+'_s'+str(r['training_seed'])]=dict(sha256=sha,bytes=n,committed_relative_path=rp,git_blob=oid)
 out=dict(passed=True,all_reference_bytes_already_committed=True,reference_jobs=45,reference_files=len(portable),files=portable,parent_and_reference_models=models)
 (ROOT/'reference_reuse_portability.json').write_text(json.dumps(out,indent=2))
 print('ISSUE19_REFERENCES_PORTABLE_PASS',len(portable),len(models),flush=True)
if __name__=='__main__':main()
