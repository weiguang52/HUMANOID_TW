"""Try remote Linux git push with memory-only credentials and bounded network wait."""
import getpass,os,subprocess,signal,tempfile,json,time
from pathlib import Path
from issue20_common import ROOT,save
plan=json.loads((ROOT/'publication/api_blob_upload_plan.json').read_text());assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['final'];token=getpass.getpass('GitHub token (memory only): ');pub=ROOT/'publication';tmp=pub/'git_tmp';tmp.mkdir(exist_ok=True)
helper=tmp/'askpass.py';helper.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("weiguang52" if "username" in sys.argv[1].lower() else os.environ["ISSUE20_GITHUB_TOKEN"])\n');helper.chmod(0o700);env=os.environ.copy()
for key in list(env):
 if key.lower() in ['http_proxy','https_proxy','all_proxy'] or key.startswith('GIT_TRACE') or key=='GIT_CURL_VERBOSE':env.pop(key,None)
env.update(GIT_ASKPASS=str(helper),GIT_TERMINAL_PROMPT='0',ISSUE20_GITHUB_TOKEN=token,TMPDIR=str(tmp));a=['git','-c','credential.helper=','-c','http.proxy=','-c','https.proxy=','-c','http.version=HTTP/1.1','-c','gc.auto=0','push','--porcelain','https://github.com/weiguang52/HUMANOID_TW.git','HEAD:refs/heads/supermimic/tw-locomotion-20261108'];started=time.time();child=subprocess.Popen(a,env=env,start_new_session=True);timed_out=False
try:code=child.wait(timeout=180)
except subprocess.TimeoutExpired:
 timed_out=True;os.killpg(child.pid,signal.SIGTERM)
 try:code=child.wait(timeout=10)
 except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
save('publication/native_push_receipt.json',{'passed':code==0,'returncode':code,'timed_out':timed_out,'commit':plan['final'],'driver_pid':os.getpid(),'child_pid':child.pid,'elapsed_seconds':time.time()-started,'credentials_persisted':False,'clean_HTTPS_remote':True,'proxy_cleared':True});print('REMOTE_NATIVE_GIT_PUSH_TERMINAL',code,timed_out,flush=True)
