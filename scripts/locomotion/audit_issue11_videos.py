"""Short exact-state failure-window replay QA, not20s resimulation QA."""
import json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
from scipy.ndimage import binary_opening,label,find_objects
ROOT=Path('/root/gpufree-data/tw_stage1/issue11_20261110');frames=ROOT/'video_review_frames';frames.mkdir(exist_ok=True);checks={};pictures=[]
for mode in [1,2,3,4]:
 selection=json.loads((ROOT/f'representative_mode{mode}.json').read_text())
 for view in ['front','side']:
  output=ROOT/f'mode{mode}_{view}.json';d=json.loads(output.read_text());p=Path(d['video']);assert p.exists() and not d['physics_integrated']
  assert d['source_environment']==selection['env_id'] and d['source_model']==selection['model'] and d['view']==view
  log=output.with_suffix('.log').read_text();assert 'Colored USD integrity: 1876 original attributes unchanged' in log
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(p)]))['streams'][0]
  n=int(probe['nb_frames']);assert n==d['frames'] and n<=51 and n>0 and probe['r_frame_rate']=='50/1' and probe['codec_name']=='h264' and (probe['width'],probe['height'])==(960,720)
  samples=[]
  for frame in sorted(set([0,n//2,n-1])):
   out=frames/(p.stem+f'_frame{frame}.png');subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-y','-i',str(p),'-vf',f'select=eq(n\\,{frame})','-frames:v','1',str(out)],check=True)
   im=Image.open(out).convert('RGB');arr=np.asarray(im).astype('int16');assert arr.max()>0
   mask=(arr.max(-1)-arr.min(-1)>35)&(arr.max(-1)>70);mask[:42]=False;mask=binary_opening(mask,structure=np.ones((3,3)));labels,_=label(mask)
   for component,region in enumerate(find_objects(labels),1):
    h=region[0].stop-region[0].start;w=region[1].stop-region[1].start
    if h<=5 and w>=20:patch=mask[region];patch[labels[region]==component]=False
   yy,xx=np.where(mask);assert len(xx)>100,(p,frame,'missing colored body');bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())];assert bbox[0]>10 and bbox[1]>42 and bbox[2]<950 and bbox[3]<710,(p,frame,bbox)
   samples.append({'frame':frame,'bbox':bbox});pictures.append((p.stem,frame,out))
  checks[p.name]={'probe':probe,'samples':samples,'exact_recorded_states':True,'physics_integrated':False,'source_npz_sha256':d['source_npz_sha256']}
(ROOT/'video_quality_audit.json').write_text(json.dumps({'passed':True,'videos':checks,'frames_sampled':len(pictures)},indent=2))
canvas=Image.new('RGB',(960,270*((len(pictures)+2)//3)),'white');draw=ImageDraw.Draw(canvas)
for i,(name,f,path) in enumerate(pictures):
 im=Image.open(path);im.thumbnail((320,240));x=i%3*320;y=i//3*270;canvas.paste(im,(x,y));draw.text((x+2,y+242),name+f' frame{f}',fill='black')
canvas.save(ROOT/'issue11_video_contact_sheet.png');print('VIDEO_QA_PASS',len(checks),'exact-state replays;',len(pictures),'sampled frames')
