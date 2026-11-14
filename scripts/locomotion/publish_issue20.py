import json,pathlib,subprocess,getpass,urllib.request,urllib.error,time,threading,concurrent.futures,base64,hashlib,datetime,re,os,traceback
ROOT=pathlib.Path('/root/tw_stage1/issue20_20261114/publication');PLAN=json.loads((ROOT/'api_blob_upload_plan.json').read_text());token=getpass.getpass('GitHub token (memory only): ');lock=threading.Lock();rate_lock=threading.Lock();next_request=0.0
records=json.loads((ROOT/'api_verified_blobs.json').read_text()) if (ROOT/'api_verified_blobs.json').exists() else {};active={};started=time.time()
def write(path,value):
 p=ROOT/path;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def state(status,**kw):
 write('api_publication_progress.json',dict(status=status,driver_pid=os.getpid(),completed_blobs=len(records),total_blobs=len(PLAN['blobs']),completed_bytes=sum(v['bytes'] for v in records.values()),active=dict(active),started_at=started,updated_at=time.time(),**kw))
def api(path,data=None,method=None):
 global next_request
 if data is not None:
  with rate_lock:
   delay=max(0,next_request-time.monotonic())
   if delay:time.sleep(delay)
   next_request=time.monotonic()+1.0
 body=None if data is None else json.dumps(data).encode()
 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'Codex','Content-Type':'application/json','Cache-Control':'no-cache'}
 request=urllib.request.Request('https://api.github.com/repos/weiguang52/HUMANOID_TW/'+path,data=body,method=method,headers=headers)
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(request,timeout=300) as response:return json.load(response)
def upload(item):
 oid=item['oid']
 if oid in records:return
 data=subprocess.check_output(['git','cat-file','blob',oid]);assert len(data)==item['bytes'];assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==oid
 with lock:active[oid]={'bytes':len(data),'started_at':time.time()};state('uploading_git_blobs')
 payload={'encoding':'base64','content':base64.b64encode(data).decode()}
 for attempt in range(4):
  try:
   result=api('git/blobs',payload,'POST');assert result['sha']==oid
   with lock:
    records[oid]={'sha':result['sha'],'bytes':len(data),'url':result['url']};active.pop(oid,None);write('api_verified_blobs.json',records);state('uploading_git_blobs')
   print('BLOB_VERIFIED',len(records),oid,len(data),flush=True);return
  except Exception as error:
   if attempt==3:raise
   wait=30*(2**attempt)
   if isinstance(error,urllib.error.HTTPError):
    text=error.read().decode(errors='replace');print('API_RETRY',oid,error.code,text[:350],flush=True)
    if error.code not in [401,403,408,429,500,502,503,504]:raise
    wait=max(wait,int(error.headers.get('Retry-After','0')))
   else:print('API_RETRY',oid,type(error).__name__,flush=True)
   with lock:active[oid].update(retry=attempt+1,wait_seconds=wait,error=type(error).__name__);state('uploading_with_bounded_retry')
   time.sleep(wait)
class AlreadyPublished(Exception):pass
try:
 remote=api('git/ref/heads/supermimic/tw-locomotion-20261108')
 if remote['object']['sha']==PLAN['final']:raise AlreadyPublished()
 assert remote['object']['sha']==PLAN['base']
 state('checking_one_blob')
 pending=[x for x in PLAN['blobs'] if x['oid'] not in records]
 if pending:
  probe=min(pending,key=lambda x:abs(x['bytes']-2_000_000));upload(probe)
 pending=sorted([x for x in PLAN['blobs'] if x['oid'] not in records],key=lambda x:-x['bytes'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(upload,item) for item in pending]
  for future in concurrent.futures.as_completed(futures):future.result()
 assert len(records)==len(PLAN['blobs'])
 state('creating_exact_git_tree')
 paths=subprocess.check_output(['git','diff','--name-only','-z',PLAN['base'],PLAN['final']]).decode().split('\0');paths=set(p for p in paths if p);entries=[]
 for raw in subprocess.check_output(['git','ls-tree','-rz',PLAN['final']]).split(b'\0'):
  if not raw:continue
  head,path=raw.split(b'\t',1);path=path.decode()
  if path in paths:
   mode,kind,oid=head.decode().split();entries.append({'path':path,'mode':mode,'type':kind,'sha':oid})
 assert len(entries)==len(subprocess.check_output(['git','diff','--name-only',PLAN['base'],PLAN['final']],text=True).splitlines())
 tree_oids=[]
 for row in subprocess.check_output(['git','ls-tree','-rt',PLAN['final']],text=True).splitlines():
  meta,path=row.split('\t',1);mode,kind,oid=meta.split()
  if kind=='tree':tree_oids.append(oid)
 new_objects={x.split()[0] for x in json.loads((ROOT/'new_git_objects.json').read_text())['objects']}
 tree_oids=[x for x in reversed(tree_oids) if x in new_objects]+[PLAN['tree']]
 assert len(set(tree_oids))>0
 verified_trees={line.split()[1] for line in ((ROOT/'api_publish.log').read_text() if (ROOT/'api_publish.log').exists() else '').splitlines() if line.startswith('TREE_VERIFIED ')}
 for oid in dict.fromkeys(tree_oids):
  if oid in verified_trees:continue
  print('TREE_START',oid,flush=True)
  rows=[]
  for rawrow in subprocess.check_output(['git','ls-tree','-z',oid]).split(b'\0'):
   if not rawrow:continue
   meta,path=rawrow.split(b'\t');mode,kind,sha=meta.decode().split()
   rows.append({'path':path.decode(),'mode':mode,'type':kind,'sha':sha})
  for attempt in range(4):
   try:
    tree=api('git/trees',{'tree':rows},'POST');assert tree['sha']==oid;break
   except urllib.error.HTTPError as e:
    print('TREE_API_ERROR',oid,e.code,e.read().decode(errors='replace')[:3000],flush=True)
    if e.code not in [422,502,503,504] or attempt==3:raise
    time.sleep(10*(attempt+1))
  print('TREE_VERIFIED',oid,len(rows),flush=True)
 write('api_exact_tree_receipt.json',tree)
 raw=subprocess.check_output(['git','cat-file','commit',PLAN['final']]).decode();head,message=raw.split('\n\n',1);headers={}
 for line in head.splitlines():
  key,value=line.split(' ',1);headers.setdefault(key,[]).append(value)
 assert set(headers)=={'tree','parent','author','committer'} and headers['tree']==[PLAN['tree']] and headers['parent']==[PLAN['base']]
 def person(value):
  m=re.fullmatch(r'(.*) <([^>]+)> (\d+) ([+-]\d{4})',value);assert m
  name,email,epoch,offset=m.groups();minutes=(int(offset[1:3])*60+int(offset[3:]))*(1 if offset[0]=='+' else -1);tz=datetime.timezone(datetime.timedelta(minutes=minutes));return {'name':name,'email':email,'date':datetime.datetime.fromtimestamp(int(epoch),tz).isoformat()}
 state('creating_exact_commit')
 created=api('git/commits',{'message':message,'tree':PLAN['tree'],'parents':[PLAN['base']],'author':person(headers['author'][0]),'committer':person(headers['committer'][0])},'POST');assert created['sha']==PLAN['final'];write('api_exact_commit_receipt.json',created)
 assert api('git/ref/heads/supermimic/tw-locomotion-20261108')['object']['sha']==PLAN['base']
 api('git/refs/heads/supermimic/tw-locomotion-20261108',{'sha':PLAN['final'],'force':False},'PATCH')
 current=api('git/ref/heads/supermimic/tw-locomotion-20261108');assert current['object']['sha']==PLAN['final']
 write('push_receipt.json',{'passed':True,'expected_commit':PLAN['final'],'ref':current,'method':'remote GitHub Git API; exact blob, tree and original commit SHA verified','force':False,'temporary_branch_created':False})
 state('remote_original_ref_verified_full_audit_pending',final_commit=PLAN['final']);print('REMOTE_ORIGINAL_COMMIT_VERIFIED',PLAN['final'],flush=True)
except AlreadyPublished:
 state('remote_original_ref_verified_full_audit_pending',final_commit=PLAN['final']);print('ALREADY_PUBLISHED_EXACT_REF',PLAN['final'],flush=True)
except BaseException:
 with lock:state('failed',error=traceback.format_exc())
 raise
