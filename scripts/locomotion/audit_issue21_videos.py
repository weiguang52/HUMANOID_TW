"""Verify codec, color, body framing, provenance and extract actual-review frames."""
import cv2,json,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from issue21_common import ROOT,sha,save

def main():
 plan=json.loads((ROOT/'video_plan.json').read_text());assert len(plan)==4;frames_dir=ROOT/'video_review_frames';frames_dir.mkdir(exist_ok=True);sheets_dir=ROOT/'video_contact_sheets';sheets_dir.mkdir(exist_ok=True);checks=[];manual=[]
 for row in plan:
  s=json.loads((ROOT/('replay_%s_k%d.json'%(row['command'],row['phase']))).read_text());z=np.load(s['trajectory'])['root_and_joints'];names=s['joint_names'];arms=[i+7 for i,n in enumerate(names) if any(v in n for v in ['shoulder','elbow','wrist'])];arm=int(np.abs(z[:,arms]-z[0,arms]).sum(1).argmax());q=z[:,3:7];yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)));turn=int(abs(np.diff(yaw,prepend=yaw[0])).argmax());selected=list(dict.fromkeys([arm,turn,999,0,500]))[:3];samples=sorted(set([0,250,500,750,999]+selected));tiles=[]
  for view in ['front','side']:
   p=Path(row['views'][view]['path']);assert sha(p)==row['views'][view]['sha256'];cap=cv2.VideoCapture(str(p));assert cap.isOpened();count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));assert (count,w,h)==(1000,960,720) and abs(fps-50)<1e-6
   for f in samples:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);region=rgb[100:675,:730];chroma=region.max(2).astype(int)-region.min(2).astype(int);ys,xs=np.where(chroma>40);assert len(xs)>100 and xs.min()>5 and xs.max()<725 and ys.min()>5 and ys.max()<570,(p,f,'color/body crop');tile=Image.fromarray(rgb).resize((320,240));canvas=Image.new('RGB',(320,260),'white');canvas.paste(tile,(0,0));ImageDraw.Draw(canvas).text((4,243),'%s k%d %s frame%d'%(row['command'],row['phase'],view,f),fill='black');tiles.append(canvas)
    if f in selected:
     target=frames_dir/('%s_k%d_%s_frame%d.png'%(row['command'],row['phase'],view,f));Image.fromarray(rgb).save(target);manual.append({'path':str(target.relative_to(ROOT)),'sha256':sha(target),'width':960,'height':720,'frame':f,'view':view})
   cap.release();checks.append({'path':str(p),'frames':count,'fps':fps,'size':[w,h],'sampled_frames':samples,'color_and_full_body_pass':True})
  sheet=Image.new('RGB',(320*len(samples),520),'white')
  for i,tile in enumerate(tiles):sheet.paste(tile,((i%len(samples))*320,(i//len(samples))*260))
  target=sheets_dir/('%s_k%d.png'%(row['command'],row['phase']));sheet.save(target)
 save('video_quality_audit.json',{'passed':True,'clips':checks,'total_clips':8,'full_resolution_review_plan':manual,'actual_review_separate':True});print('ISSUE21_VIDEO_AUTOMATED_QA_PASS',len(checks),len(manual),flush=True)
if __name__=='__main__':main()
