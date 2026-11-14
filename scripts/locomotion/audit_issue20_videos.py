"""Check encoding, seven time/motion samples per view, and paired-video timing."""
from pathlib import Path
import json,subprocess,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_opening,label,find_objects
from issue20_common import ROOT,sha,save

def probe(p,width=960):
 v=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(p)]))['streams'][0];assert v['codec_name']=='h264' and int(v['nb_frames'])==1000 and v['r_frame_rate']=='50/1' and (v['width'],v['height'])==(width,720) and abs(float(v['duration'])-20)<1e-6;return v

def main():
 assert (ROOT/'videos_complete').exists();plan=json.loads((ROOT/'video_plan.json').read_text());assert len(plan)==2;frames=ROOT/'video_review_frames';frames.mkdir(exist_ok=True);sheets=ROOT/'video_contact_sheets';sheets.mkdir(exist_ok=True);checks={}
 for row in plan:
  s=json.loads((ROOT/f'replay_selection_k{row["phase"]}.json').read_text());z=np.load(s['trajectory']);v=z['root_and_joints'];yaw=np.load(s['yaw_file'])['heading_yaw'][:,0];ids=[i for i,n in enumerate(s['joint_names']) if any(x in n for x in ['shoulder','elbow','wrist'])];arm=np.r_[0,np.linalg.norm(np.diff(v[:,7:][:,ids],axis=0),axis=1)];turn=np.r_[0,abs(np.arctan2(np.sin(np.diff(yaw)),np.cos(np.diff(yaw))))];arm_frame=int(arm.argmax());turn_frame=int(turn.argmax());sample={0,250,500,750,999,arm_frame,turn_frame}
  for score in [arm,turn]:
   if len(sample)<7:sample.add(max((i for i in range(1000) if i not in sample),key=lambda i:(score[i],-i)))
  sample=sorted(sample);assert len(sample)==7;pictures=[]
  for view in ['front','side','overhead']:
   p=Path(row['views'][view]['path']);assert sha(p)==row['views'][view]['sha256'];pr=probe(p);examined=[]
   for i in sample:
    dest=frames/(p.stem+f'_frame{i}.png');subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-y','-i',str(p),'-vf',f'select=eq(n\\,{i})','-frames:v','1',str(dest)],check=True);im=Image.open(dest).convert('RGB');a=np.asarray(im).astype('int16');mask=(a.max(-1)-a.min(-1)>35)&(a.max(-1)>70);mask[:78]=False;mask[680:]=False;mask[78:243,740:953]=False;mask=binary_opening(mask,structure=np.ones((3,3)));components,_=label(mask)
    for component,region in enumerate(find_objects(components),1):
     h=region[0].stop-region[0].start;w=region[1].stop-region[1].start
     if (h<=5 and w>=20) or np.count_nonzero(components[region]==component)<50:patch=mask[region];patch[components[region]==component]=False
    yy,xx=np.where(mask);assert len(xx)>100,(p,i);box=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())];assert box[0]>10 and box[1]>78 and box[2]<950 and box[3]<670,(p,i,box);examined.append({'frame':i,'bbox':box,'sha256':sha(dest)});pictures.append((view,i,dest))
   checks[p.name]={'probe':pr,'sampled_frames':examined,'arm_motion_frame':arm_frame,'turn_frame':turn_frame,'manual_review_frames':[arm_frame,turn_frame,999]}
  sheet=Image.new('RGB',(1680,630),'white');draw=ImageDraw.Draw(sheet)
  for j,(view,i,p) in enumerate(pictures):
   im=Image.open(p);im.thumbnail((240,180));x=(j%7)*240;y=(j//7)*210;sheet.paste(im,(x,y));draw.text((x+2,y+182),f'k={row["phase"]} {view} f{i}',fill='black')
  sheet.save(sheets/f'phase{row["phase"]}.png')
 paired=json.loads((ROOT/'paired_overhead_video.json').read_text());assert sha(paired['video'])==paired['sha256'];pr=probe(paired['video'],1920)
 save('video_quality_audit.json',{'passed':True,'videos':checks,'paired_overhead_probe':pr,'single_view_clips':6,'paired_clips':1,'single_view_sampled_frames':42,'contact_sheets':2,'manual_attachment_full_frame_and_plot_review_required':True});print('ISSUE20_VIDEO_QA_PASS',len(checks),'single clips plus paired overhead',flush=True)
if __name__=='__main__':main()
