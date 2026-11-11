"""Encoding, full-body/color and explicit push-response frame sampling; manual review remains separate."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_opening,label,find_objects
from issue14_common import ROOT,sha
def main():
 assert (ROOT/"videos_complete").exists();plan=json.loads((ROOT/"video_plan.json").read_text());assert len(plan)==3
 frames=ROOT/"video_review_frames";frames.mkdir(exist_ok=True);sheets=ROOT/"video_contact_sheets";sheets.mkdir(exist_ok=True)
 checks={}
 for example in plan:
  base_frames={0,250,500,750,999}
  if example["push_seconds"]:
   response=min(999,round(example["push_seconds"][0]/.02)-1+10)
   push_frame=round(example["push_seconds"][0]/.02)-1
   response=min((i for i in range(max(0,push_frame),min(1000,push_frame+50)) if i not in base_frames),key=lambda i:(abs(i-response),i))
  else:response=400
  sample=sorted(base_frames|{response});assert len(sample)==6
  pictures=[]
  for view in ["front","side"]:
   p=Path(example["views"][view]["path"]);assert sha(p)==example["views"][view]["sha256"]
   probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_name,width,height,nb_frames,r_frame_rate,duration","-of","json",str(p)]))["streams"][0]
   assert probe["codec_name"]=="h264" and int(probe["nb_frames"])==1000 and probe["r_frame_rate"]=="50/1"
   assert (probe["width"],probe["height"])==(960,720) and abs(float(probe["duration"])-20)<1e-6
   examined=[]
   for frame in sample:
    out=frames/(p.stem+f"_frame{frame}.png")
    subprocess.run(["ffmpeg","-nostdin","-v","error","-threads","1","-y","-i",str(p),"-vf",f"select=eq(n\\,{frame})","-frames:v","1",str(out)],check=True)
    im=Image.open(out).convert("RGB");a=np.asarray(im).astype("int16")
    mask=(a.max(-1)-a.min(-1)>35)&(a.max(-1)>70);mask[:70]=False;mask[680:]=False
    mask=binary_opening(mask,structure=np.ones((3,3)));labels,_=label(mask)
    for component,region in enumerate(find_objects(labels),1):
     h=region[0].stop-region[0].start;w=region[1].stop-region[1].start
     if h<=5 and w>=20:
      patch=mask[region];patch[labels[region]==component]=False
    yy,xx=np.where(mask);assert len(xx)>100,(p,frame)
    bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
    assert bbox[0]>10 and bbox[1]>70 and bbox[2]<950 and bbox[3]<670,(p,frame,bbox)
    examined.append({"frame":frame,"bbox":bbox,"sha256":sha(out),"push_response_sample":frame==response and bool(example["push_seconds"])})
    pictures.append((view,frame,out))
   checks[p.name]={"probe":probe,"sampled_frames":examined,"push_response_frame":response if example["push_seconds"] else None,"motion_review_frame":response}
  sheet=Image.new("RGB",(1440,420),"white");draw=ImageDraw.Draw(sheet)
  for i,(view,frame,p) in enumerate(pictures):
   im=Image.open(p);im.thumbnail((240,180));x=(i%6)*240;y=(i//6)*210;sheet.paste(im,(x,y))
   draw.text((x+2,y+182),f"{example['group']} {view} f{frame}",fill="black")
  sheet.save(sheets/(example["group"]+".png"))
 assert len(checks)==6
 (ROOT/"video_quality_audit.json").write_text(json.dumps({"passed":True,"videos":checks,"encoding_frames":6000,
  "sampled_frames":36,"contact_sheets":3,"manual_attachment_plot_review_required":True},indent=2))
 print("ISSUE14_VIDEO_QA_PASS6CLIPS36SAMPLES",flush=True)
if __name__=="__main__":main()
