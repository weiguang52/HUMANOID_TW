"""Read-only publication preflight; credentials remain in foreground process memory."""
import base64,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue16_20261111')
BASE='770014ee9a4393da6dff34eced89a04d83e788dd'
BRANCH='supermimic/tw-locomotion-20261108'
def main():
 token=sys.stdin.readline().strip();assert token
 headers={'Authorization':'Bearer '+token,'User-Agent':'TW-issue16-readiness','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
 def get(path):
  req=urllib.request.Request('https://api.github.com/repos/weiguang52/HUMANOID_TW'+path,headers=headers)
  with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
 out={'checked_unix':time.time(),'read_only':True}
 ref=get('/git/ref/heads/'+BRANCH);out['remote_branch_sha']=ref['object']['sha'];out['expected_base_sha']=BASE;out['remote_still_at_frozen_base']=out['remote_branch_sha']==BASE
 issue=get('/issues/16');(ROOT/'issue16_prepublication_readback.json').write_text(json.dumps(issue,indent=2)+'\n')
 old=json.loads((ROOT/'issue_source.json').read_text())
 out['issue_body_unchanged']=issue['body']==old['body'];out['issue_title_unchanged']=issue['title']==old['title']
 env=dict(os.environ)
 env.update(GIT_TERMINAL_PROMPT='0',GIT_ASKPASS='/bin/false',GIT_CONFIG_COUNT='3',
  GIT_CONFIG_KEY_0='http.extraHeader',GIT_CONFIG_VALUE_0='Authorization: Basic '+base64.b64encode(('x-access-token:'+token).encode()).decode(),
  GIT_CONFIG_KEY_1='credential.helper',GIT_CONFIG_VALUE_1='',GIT_CONFIG_KEY_2='http.version',GIT_CONFIG_VALUE_2='HTTP/1.1')
 attempts=[]
 for mode in ['current_environment','proxy_cleared']:
  current=dict(env)
  if mode=='proxy_cleared':
   for k in list(current):
    if k.lower() in ['http_proxy','https_proxy','all_proxy']:current.pop(k,None)
   current['GIT_CONFIG_COUNT']='4';current['GIT_CONFIG_KEY_3']='http.proxy';current['GIT_CONFIG_VALUE_3']=''
  try:
   r=subprocess.run(['git','ls-remote','https://github.com/weiguang52/HUMANOID_TW.git','refs/heads/'+BRANCH],env=current,capture_output=True,timeout=25)
   text=r.stdout.decode(errors='replace').strip()
   attempts.append(dict(mode=mode,returncode=r.returncode,matching_sha=text.split()[0] if r.returncode==0 and text else None,error=r.stderr.decode(errors='replace')[-700:].replace(token,'[REDACTED]')))
   if r.returncode==0:break
  except subprocess.TimeoutExpired:attempts.append(dict(mode=mode,timeout_seconds=25))
 out['native_git_read_attempts']=attempts
 (ROOT/'publication_network_check.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
