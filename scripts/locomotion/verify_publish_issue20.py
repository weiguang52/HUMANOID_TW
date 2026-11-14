"""Verify published exact Git objects, clean own upload ref, then reply to issue19."""
import json,pathlib,subprocess,getpass,urllib.request,urllib.error,gzip,time,hashlib
ROOT=pathlib.Path('/root/tw_stage1/issue20_20261114/publication');REPO=pathlib.Path.cwd();PLAN=json.loads((ROOT/'api_blob_upload_plan.json').read_text());FINAL=PLAN['final'];BASE=PLAN['base'];TREE=PLAN['tree'];BRANCH='supermimic/tw-locomotion-20261108'

token=getpass.getpass('GitHub token (memory only): ');opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def api(path,data=None,method=None):
 body=None if data is None else json.dumps(data,ensure_ascii=False).encode()
 for attempt in range(3):
  try:
   req=urllib.request.Request('https://api.github.com/repos/weiguang52/HUMANOID_TW/'+path,data=body,method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'Codex','Content-Type':'application/json','Accept-Encoding':'gzip','Cache-Control':'no-cache'})
   with opener.open(req,timeout=60) as response:
    raw=response.read()
    if response.headers.get('Content-Encoding')=='gzip':raw=gzip.decompress(raw)
   return json.loads(raw) if raw else None
  except urllib.error.HTTPError:raise
  except Exception:
   if attempt==2:raise
   time.sleep(3)
def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def entries(tree):return {r['path']:r for r in tree['tree']}
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==FINAL
ref=api('git/ref/heads/'+BRANCH);assert ref['object']['sha']==FINAL
commit=api('git/commits/'+FINAL);assert commit['tree']['sha']==TREE and [x['sha'] for x in commit['parents']]==[BASE]
root=entries(api('git/trees/'+TREE));validation=entries(api('git/trees/'+root['validation_artifacts']['sha']));archive_node=validation['stage1_issue20_20261114'];remote_archive=api('git/trees/'+archive_node['sha']+'?recursive=1');assert not remote_archive.get('truncated')
remote={x['path']:(x['mode'],x['sha']) for x in remote_archive['tree'] if x['type']=='blob'};local={}
for raw in subprocess.check_output(['git','ls-tree','-rz',FINAL+':validation_artifacts/stage1_issue20_20261114']).split(b'\0'):
 if not raw:continue
 head,path=raw.split(b'\t',1);mode,kind,oid=head.decode().split();assert kind=='blob';local[path.decode()]=(mode,oid)
expected_archive=json.loads((ROOT/'archive_receipt.json').read_text())['archive_files']
assert remote==local and len(local)==expected_archive
tree_cache={TREE:root}
def lookup(path):
 node=root
 for part in path.split('/'):
  row=node[part]
  if row['type']=='tree':
   if row['sha'] not in tree_cache:tree_cache[row['sha']]=entries(api('git/trees/'+row['sha']))
   node=tree_cache[row['sha']]
 return row
changed=[x for x in subprocess.check_output(['git','diff','--name-only','-z',BASE,FINAL]).decode().split('\0') if x];source_count=0
for path in changed:
 if path.startswith('validation_artifacts/stage1_issue20_20261114/'):continue
 localrow=subprocess.check_output(['git','ls-tree',FINAL,'--',path],text=True).strip();head,_=localrow.split('\t');mode,kind,oid=head.split();row=lookup(path)
 assert (row['mode'],row['type'],row['sha'])==(mode,kind,oid);source_count+=1
assert source_count==len(changed)-expected_archive and source_count>0
save('github_exact_ref_commit.json',{'ref':ref,'commit':commit});save('github_archive_tree.json',remote_archive)
assert api('git/ref/heads/'+BRANCH)['object']['sha']==FINAL
subprocess.run(['git','update-ref','refs/remotes/origin/'+BRANCH,FINAL],check=True)
latest=api('issues?state=all&sort=created&direction=desc&per_page=20');latest_issues=[x for x in latest if 'pull_request' not in x];current=api('issues/20');comments=api('issues/20/comments?per_page=100');snapshot=json.loads((ROOT.parent/'issue_snapshot.json').read_text());reviewed=json.loads((ROOT.parent/'issue_scope_recheck.json').read_text())
save('latest_issue_scope_check.json',{'latest_issue_number':latest_issues[0]['number'],'issue20':current,'comments':comments,'frozen_issue20_body_unchanged':current['body']==snapshot['issue']['body']})
assert current['body']==snapshot['issue']['body'],'Issue20 body changed; review before completion'
body=(ROOT/'final_issue_reply.md').read_text();reviewed_ids={c['id'] for c in reviewed['comments']};unknown=[c for c in comments if c['id'] not in reviewed_ids and c['body']!=body]
assert not unknown,'New issue20 replies require scope review before final delivery'
body=(ROOT/'final_issue_reply.md').read_text()
comments=api('issues/20/comments?per_page=100');same=[c for c in comments if c['body']==body]
comment=same[0] if same else api('issues/20/comments',{'body':body},'POST')
check=api('issues/comments/'+str(comment['id']));assert check['body']==body
save('github_issue_reply_receipt.json',{'passed':True,'url':check['html_url'],'id':check['id'],'body_sha256':hashlib.sha256(body.encode()).hexdigest(),'readback':check})
audit={'passed':True,'commit':FINAL,'parent':BASE,'tree':TREE,'archive_blob_ids_verified':len(local),'code_and_document_blob_ids_verified':source_count,'temporary_branch_created':False,'issue_reply_readback_verified':True,'issue_reply_url':check['html_url'],'notion_final_delivery_pending':True,'verified_at':time.time()};save('github_publication_audit.json',audit)
print('GITHUB_PUBLICATION_ALL_OBJECTS_VERIFIED',FINAL,len(local),source_count,check['html_url'],flush=True)
