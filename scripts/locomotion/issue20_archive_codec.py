"""Lossless deterministic gzip/chunks below GitHub's per-blob size ceiling."""
from pathlib import Path
import gzip,hashlib,io,os,shutil,tempfile
CHUNK=4*1024*1024
TEXT={'.csv','.usd','.usda','.json','.jsonl','.yaml','.yml','.log','.md','.txt'}
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def contained(root,relative):
 p=Path(relative);assert not p.is_absolute() and '..' not in p.parts,relative
 target=(Path(root)/p).resolve();assert target.is_relative_to(Path(root).resolve()),relative;return target
class PartsReader(io.RawIOBase):
 def __init__(self,root,parts):super().__init__();self.paths=iter(contained(root,x['path']) for x in parts);self.current=None
 def readable(self):return True
 def readinto(self,b):
  while True:
   if self.current is None:
    p=next(self.paths,None)
    if p is None:return 0
    self.current=p.open('rb')
   n=self.current.readinto(b)
   if n:return n
   self.current.close();self.current=None
 def close(self):
  if self.current is not None:self.current.close()
  super().close()
def decoded(root,row):
 for part in row['parts']:
  p=contained(root,part['path']);assert p.stat().st_size==part['bytes'] and digest(p)==part['sha256'],p
 stream=io.BufferedReader(PartsReader(root,row['parts']))
 return gzip.GzipFile(fileobj=stream,mode='rb') if row['encoding']=='gzip' else stream

def encode(source,destination,relative,tmpdir,chunk_bytes=CHUNK):
 source=Path(source);target=contained(destination,relative);size=source.stat().st_size;rawsha=digest(source)
 if size<=chunk_bytes or (source.suffix.lower()=='.mp4' and size<=90*1024*1024):
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert digest(target)==rawsha;return None
 encoding='gzip' if source.suffix.lower() in TEXT else 'raw';temp=None;payload=source
 try:
  if encoding=='gzip':
   fd,temp=tempfile.mkstemp(prefix='issue20-codec-',suffix='.gz',dir=tmpdir);os.close(fd);payload=Path(temp)
   with source.open('rb') as src,payload.open('wb') as dst:
    with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0) as gz:shutil.copyfileobj(src,gz,1048576)
  parts=[];folder=contained(destination,'.chunks/'+relative);folder.mkdir(parents=True,exist_ok=True)
  with payload.open('rb') as f:
   i=0
   while True:
    b=f.read(chunk_bytes)
    if not b:break
    p=folder/f'part{i:06d}.bin';p.write_bytes(b);parts.append({'path':str(p.relative_to(destination)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});i+=1
  row={'path':relative,'bytes':size,'sha256':rawsha,'encoding':encoding,'parts':parts};h=hashlib.sha256();n=0
  with decoded(destination,row) as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b);n+=len(b)
  assert n==size and h.hexdigest()==rawsha,(relative,n,size);return row
 finally:
  if temp is not None:Path(temp).unlink(missing_ok=True)

def restore(destination,row,output):
 target=contained(output,row['path']);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert target.stat().st_size==row['bytes'] and digest(target)==row['sha256'];return
 temp=target.with_name(target.name+'.issue20-restoring');assert not temp.exists();h=hashlib.sha256();n=0
 try:
  with decoded(destination,row) as src,temp.open('xb') as dst:
   for b in iter(lambda:src.read(1048576),b''):dst.write(b);h.update(b);n+=len(b)
  assert n==row['bytes'] and h.hexdigest()==row['sha256'];temp.replace(target)
 finally:temp.unlink(missing_ok=True)
