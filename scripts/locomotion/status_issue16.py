"""Read only actual issue16 process liveness, completed counts and evaluation-only ETA."""
import json,time,statistics
from pathlib import Path
from issue16_common import ROOT
def alive(pid):
 p=Path('/proc')/str(pid)
 try:return p.exists() and (p/'stat').read_text().split()[2]!='Z'
 except (FileNotFoundError,ProcessLookupError):return False
def main():
 jobs=json.loads((ROOT/'formal_jobs.json').read_text());p=ROOT/'formal_job_audits.json'
 audits=json.loads(p.read_text()) if p.exists() else []
 driver=json.loads((ROOT/'formal_driver_process.json').read_text());p=ROOT/'progress.json';progress=json.loads(p.read_text()) if p.exists() else {}
 times=[Path(j['output']).stat().st_mtime for j in jobs if Path(j['output']).exists()]
 intervals=[b-a for a,b in zip([driver.get('started_unix',times[0] if times else time.time()),*times[:-1]],times) if b>a]
 median=statistics.median(intervals[-8:]) if intervals else None
 eta=max(0,median*(len(jobs)-len(audits))-(time.time()-times[-1] if times else 0)) if median is not None else None
 post=ROOT/'postprocess_driver.json';post=json.loads(post.read_text()) if post.exists() else None
 media=ROOT/'media_driver.json';media=json.loads(media.read_text()) if media.exists() else None
 out=dict(formal_completed=len(audits),formal_total=len(jobs),formal_complete=(ROOT/'formal_evaluation_complete').exists(),
  actual_formal_driver_live=alive(driver['pid']),current_phase=progress.get('phase'),current_child_pid=progress.get('pid'),actual_current_child_live=alive(progress.get('pid',-1)),
  recent_median_job_wall_seconds=median,formal_evaluation_eta_seconds=eta,ETA_excludes_analysis_media_QA_and_publication=True,
  postprocessing=None if post is None else dict(post,actual_process_live=alive(post['pid'])),media=None if media is None else dict(media,actual_process_live=alive(media['pid'])))
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
