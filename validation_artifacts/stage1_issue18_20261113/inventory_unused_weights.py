import json,re,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113');REPO=Path('/root/gpufree-data/projects/HUMANOID_TW');LOGS=REPO/'logs/rsl_rl/tw_stage1';pattern=re.compile(rb'model_[0-9]+\.pt');references=set();seen=set();count=0
for root in [REPO,Path('/root/gpufree-data/tw_stage1')]:
 for p in root.rglob('*'):
  if not p.is_file() or '.git' in p.parts or p.suffix.lower() not in {'.json','.yaml','.yml','.md','.py','.sh','.log','.txt','.patch'}:continue
  st=p.stat();key=(st.st_dev,st.st_ino)
  if key in seen:continue
  seen.add(key);count+=1;tail=b''
  with p.open('rb') as f:
   while chunk:=f.read(1024*1024):
    data=tail+chunk;references.update(x.decode() for x in pattern.findall(data));tail=data[-64:]
tracked=set(subprocess.check_output(['git','ls-files','-z']).decode().split('\0'));open_paths=set()
for p in Path('/proc').glob('[0-9]*/fd/*'):
 try:open_paths.add(str(p.resolve(strict=True)))
 except (OSError,RuntimeError):pass
candidates=[];preserved=[]
for folder in LOGS.iterdir():
 if not folder.is_dir():continue
 weights=[p for p in folder.glob('model_*.pt') if re.fullmatch(r'model_[0-9]+\.pt',p.name)];latest=max((int(p.stem.split('_')[1]) for p in weights),default=-1)
 for p in weights:
  st=p.stat();reason=[]
  if int(p.stem.split('_')[1])==latest:reason.append('latest_in_run')
  if p.name in references:reason.append('basename_referenced_in_any_scientific_or_tracked_text')
  if st.st_nlink>1:reason.append('hardlinked_or_archived')
  if str(p.relative_to(REPO)) in tracked:reason.append('Git_tracked')
  if str(p.resolve()) in open_paths:reason.append('open_by_process')
  if p.is_symlink() or not p.resolve().is_relative_to(LOGS):reason.append('path_safety')
  item=dict(path=str(p),bytes=st.st_size,nlink=st.st_nlink)
  if reason:preserved.append(dict(item,reasons=reason))
  else:candidates.append(item)
result=dict(reference_text_files_scanned=count,referenced_model_basenames=sorted(references),candidates=candidates,candidate_bytes=sum(x['bytes'] for x in candidates),preserved=preserved,authorization='User explicitly authorized deleting unused training weights; inventory only, no deletion yet.',observed_unix=time.time());(ROOT/'unused_weights_inventory.json').write_text(json.dumps(result,indent=2)+'\n');print('INVENTORY_COMPLETE',len(candidates),result['candidate_bytes'],'PRESERVED',len(preserved),'REFERENCE_TEXTS',count,flush=True)
