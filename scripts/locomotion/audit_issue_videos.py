import json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw
import numpy as np
r=Path('/root/gpufree-data/tw_stage1/issue_runs_20261108');paths=sorted(p for p in (r/'videos').glob('*.mp4') if p.name.startswith(('ft5_video_','robust08_video_')));assert len(paths)==32
frames=r/'video_review_frames';frames.mkdir(exist_ok=True)
def check(p):
 data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(p)]))['streams'][0]
 assert data['codec_name']=='h264' and int(data['nb_frames'])==1000 and data['r_frame_rate']=='50/1',(p,data)
 assert (data['width'],data['height'])==(960,720),(p,data)
 samples=[]
 for sec in [0,5,10,15,19.9]:
  out=frames/(p.stem+f'_{sec}.png')
  subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-y','-ss',str(sec),'-i',str(p),'-frames:v','1',str(out)],check=True)
  arr=np.asarray(Image.open(out).convert('RGB')).astype('int16');assert arr.max()>0,(p,sec,'black frame')
  mask=(arr.max(-1)-arr.min(-1)>35)&(arr.max(-1)>70)
  from scipy.ndimage import binary_opening
  mask=binary_opening(mask,structure=np.ones((3,3)))
  yy,xx=np.where(mask);assert len(xx)>100,(p,sec,'missing colored robot')
  bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
  assert bbox[0]>10 and bbox[1]>10 and bbox[2]<950 and bbox[3]<710,(p,sec,bbox)
  samples.append({'second':sec,'colored_pixel_bbox':bbox})
 return p.name,{'probe':data,'samples':samples,'black_frame_check':'passed','colored_framing_check':'passed'}
result=dict(ThreadPoolExecutor(max_workers=2).map(check,paths));(r/'video_quality_audit.json').write_text(json.dumps(result,indent=2));print('PASS 32 videos, 1000 frames/50fps each, 160 sampled frames nonblack with full colored framing')
pairs=[p for p in paths if p.name.startswith('robust08_video_')];canvas=Image.new('RGB',(960,270*((len(pairs)+2)//3)),'white');draw=ImageDraw.Draw(canvas)
for i,p in enumerate(pairs):
 im=Image.open(frames/(p.stem+'_10.png'));im.thumbnail((320,240));x=i%3*320;y=i//3*270;canvas.paste(im,(x,y));draw.text((x+2,y+242),p.stem.replace('robust08_video_',''),fill='black')
canvas.save(r/'robust08_video_contact_sheet.png')
