"""Verify every recorded episode and its exact front/side video duration."""
import argparse,hashlib,html,json,subprocess
from fractions import Fraction
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True)
a=p.parse_args();root=a.evaluation.resolve();report=json.loads((root/'results.json').read_text())
assert not (root/'episode_media_integrity.json').exists(), 'Preserve previous audits'
assert hashlib.sha256(Path(report['checkpoint']).read_bytes()).hexdigest()==report['checkpoint_sha256']
manifest=json.loads(Path(report['manifest']).read_text())
assert hashlib.sha256(Path(report['manifest']).read_bytes()).hexdigest()==report['manifest_sha256']
rows=report['rows'];keys=[(r['id'],r['agent']) for r in rows];assert len(keys)==len(set(keys))
ids={r['id'] for r in manifest['motions']};agents={r['agent'] for r in rows}
assert set(keys)=={(i,agent) for i in ids for agent in agents}
checks=[];body=[]
for r in rows:
 n=r['steps'];assert n>0 and n<=r['planned_steps']
 assert abs(r['duration_s']-n*.02)<1e-8
 assert r['completed_requested_window']==(n==r['planned_steps'] and not r['terminated'])
 def local(name):
  path=(root/name).resolve();assert path.is_relative_to(root);return path
 samples=json.loads(local(r['sample_file']).read_text());assert len(samples)==n
 assert [s['step'] for s in samples]==list(range(1,n+1))
 with np.load(local(r['transition_file']),allow_pickle=False) as z:
  for name,width in [('state',73),('next_state',73),('action',30),('context',34)]:
   assert z[name].shape==(n,width) and np.isfinite(z[name]).all()
  assert np.array_equal(z['next_state'][:-1],z['state'][1:])
 links=[]
 for view in ['front','side']:
  path=local(r[view+'_video'])
  stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,nb_read_frames,r_frame_rate,width,height','-of','json',str(path)],text=True))['streams'][0]
  assert stream['codec_name']=='h264' and int(stream['nb_read_frames'])==n
  assert Fraction(stream['r_frame_rate'])==50 and (stream['width'],stream['height'])==(960,640)
  subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
  checks.append(dict(id=r['id'],agent=r['agent'],view=view,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stream=stream,full_decode=True))
  links.append(f'<p>{view}</p><video controls preload="metadata" width="640" src="{html.escape(path.name)}"></video>')
 body.append(f'<h2>{html.escape(r["id"])} / {html.escape(r["agent"])}</h2><p>Actual {n}/requested {r["planned_steps"]} steps; {r["duration_s"]:.2f}s. Termination: {html.escape(str(r["termination_reasons"]))}</p>'+''.join(links))
(root/'episode_media_integrity.json').write_text(json.dumps(dict(passed=True,checkpoint_sha256=report['checkpoint_sha256'],episodes=len(rows),videos=checks,visual_review='Pending; encoding checks do not establish physical success'),indent=2))
intro='<h1>Fixed final-checkpoint physical evaluation</h1><p>First original termination stops each episode. Short clips are failures, not 20-second demonstrations. Zero means nominal PD targets, not zero torque. No promotion or full-motion acceptance is implied.</p>'
if manifest.get('synthetic_diagnostic'):intro+='<p>Synthetic static development reference; this is not a HumanML3D source or independent-motion validation.</p>'
(root/'verified_index.html').write_text('<!doctype html><meta charset="utf-8"><title>TW evaluation</title>'+intro+''.join(body))
print(json.dumps(dict(episodes=len(rows),videos=len(checks),passed=True)))
