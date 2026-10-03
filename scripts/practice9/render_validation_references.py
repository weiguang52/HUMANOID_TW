"""Render original HumanML skeletons for every clip in a validation manifest."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--source-root',type=Path,default=Path('/root/gpufree-data/datasets/practice9/humanml3d_rebuild/staging-v1/HumanML3D/new_joints'))
p.add_argument('--fps',type=float,default=20)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads(a.manifest.read_text());rows=[]
renderer=Path(__file__).with_name('visualize_humanml_source.py')
renderer_hash=hashlib.sha256(renderer.read_bytes()).hexdigest()
for motion in manifest['motions']:
 clip=motion['id']
 if Path(clip).name!=clip:raise ValueError('Invalid motion id')
 source=a.source_root/(clip+'.npy');video=a.output/(clip+'_source.mp4');meta=video.with_suffix('.json')
 sha=hashlib.sha256(source.read_bytes()).hexdigest()
 old=json.loads(meta.read_text()) if meta.exists() else {}
 if not (video.is_file() and old.get('source_sha256')==sha and old.get('fps')==a.fps and old.get('renderer_sha256')==renderer_hash):
  subprocess.run([sys.executable,str(renderer),str(source),str(video),'--fps',str(a.fps)],check=True)
  old=json.loads(meta.read_text());old['renderer_sha256']=renderer_hash
  meta.write_text(json.dumps(old,indent=2)+'\n')
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
 if info['codec_name']!='h264' or int(info['nb_frames'])!=old['shape'][0]:raise ValueError('Video frame/codec mismatch')
 numerator,denominator=map(float,info['r_frame_rate'].split('/'))
 if abs(numerator/denominator-a.fps)>1e-6:raise ValueError('Video fps mismatch')
 rows.append(dict(id=clip,video=video.name,source=str(source),source_frames=old['shape'][0],source_fps=a.fps,source_duration_s=old['duration_s'],robot_reference_duration_s=motion['frames']/manifest.get('target_fps',50),note='Original source speed; robot reference may be time-stretched. Not synchronized playback.'))
(a.output/'index.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Original NPY reference videos','','Every video uses original HumanML3D 20fps timing and fixed cameras.','Robot reference duration may differ because retargeting time-stretches motions.','Compare posture/phase; do not equate wall-clock timestamps.','','|Clip|Source seconds|Robot reference seconds|Video|','|---|---:|---:|---|']
for r in rows:lines.append(f"|{r['id']}|{r['source_duration_s']:.2f}|{r['robot_reference_duration_s']:.2f}|[video]({r['video']})|")
(a.output/'README.md').write_text('\n'.join(lines)+'\n')
print('Verified',len(rows),'reference videos',flush=True)
