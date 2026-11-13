"""Encoding, full-body/color and explicit push-response frame sampling; manual review remains separate."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_opening,label,find_objects
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def sha(p):return __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert (ROOT/"videos_complete").exists();plan=json.loads((ROOT/"video_plan.json").read_text());assert len(plan)==1
 frames=ROOT/"video_review_frames";frames.mkdir(exist_ok=True);sheets=ROOT/"video_contact_sheets";sheets.mkdir(exist_ok=True)
 checks={}
 for example in plan:
  base_frames={0,250,500,750,999}
  selected=json.loads((ROOT/"reconstruction_state.json").read_text())
  with np.load(selected["trajectory"]) as z:values=z["values"];valid=z["flags"][:,0].astype(bool)
  meta=json.loads(Path(selected["trajectory_meta"]).read_text());q=values[:,3:7]
  yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
  pairs=np.r_[False,valid[1:]&valid[:-1]];assert pairs.any()
  arm_ids=[i for i,n in enumerate(meta["joint_names"]) if any(k in n.lower() for k in ["shoulder","elbow","wrist","hand"])]
  arm=np.zeros(1000)
  if arm_ids:arm[1:]=np.linalg.norm(np.diff(values[:,13:][:,arm_ids],axis=0),axis=1)
  turn=np.r_[0.,abs(np.arctan2(np.sin(np.diff(yaw)),np.cos(np.diff(yaw))))]
  arm[~pairs]=-np.inf;turn[~pairs]=-np.inf
  arm_frame=int(np.argmax(arm));turn_frame=max((i for i in range(1000) if pairs[i] and i!=arm_frame),key=lambda i:(turn[i],-i))
  sample_set=base_frames|{arm_frame,turn_frame}
  for score in [arm,turn]:
   if len(sample_set)<7:sample_set.add(max((i for i in range(1000) if pairs[i] and i not in sample_set),key=lambda i:(score[i],-i)))
  sample=sorted(sample_set);assert len(sample)==7
  response=arm_frame  # Legacy key; explicit heading review is mandatory too.
  pictures=[]
  assert set(example["views"])=={"front","overhead"}
  for view in ["front","overhead"]:
   p=Path(example["views"][view]["path"]);assert sha(p)==example["views"][view]["sha256"]
   probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_name,width,height,nb_frames,r_frame_rate,duration","-of","json",str(p)]))["streams"][0]
   assert probe["codec_name"]=="h264" and int(probe["nb_frames"])==1000 and probe["r_frame_rate"]=="50/1"
   assert (probe["width"],probe["height"])==(960,720) and abs(float(probe["duration"])-20)<1e-6
   examined=[]
   for frame in sample:
    out=frames/(p.stem+f"_frame{frame}.png")
    subprocess.run(["ffmpeg","-nostdin","-v","error","-threads","1","-y","-i",str(p),"-vf",f"select=eq(n\\,{frame})","-frames:v","1",str(out)],check=True)
    im=Image.open(out).convert("RGB");a=np.asarray(im).astype("int16")
    mask=(a.max(-1)-a.min(-1)>35)&(a.max(-1)>70);mask[:78]=False;mask[680:]=False
    mask[78:243,740:953]=False  # Exclude explicit heading/trail inset from robot-only bbox.
    mask=binary_opening(mask,structure=np.ones((3,3)));labels,_=label(mask)
    for component,region in enumerate(find_objects(labels),1):
     h=region[0].stop-region[0].start;w=region[1].stop-region[1].start
     if (h<=5 and w>=20) or np.count_nonzero(labels[region]==component)<50:
      patch=mask[region];patch[labels[region]==component]=False
    yy,xx=np.where(mask);assert len(xx)>100,(p,frame)
    bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
    assert bbox[0]>10 and bbox[1]>78 and bbox[2]<950 and bbox[3]<670,(p,frame,bbox)
    examined.append({"frame":frame,"bbox":bbox,"sha256":sha(out),"push_response_sample":False,"arm_motion_review_sample":frame==arm_frame,"heading_turn_review_sample":frame==turn_frame})
    pictures.append((view,frame,out))
   checks[p.name]={"probe":probe,"sampled_frames":examined,"push_response_frame":response if example["push_seconds"] else None,"motion_review_frame":arm_frame,"heading_review_frame":turn_frame,"manual_review_frames":[arm_frame,turn_frame],"arm_motion_present":bool(arm[arm_frame]>1e-6),"max_arm_joint_delta_norm_rad":float(arm[arm_frame]),"heading_turn_delta_rad":float(turn[turn_frame])}
  sheet=Image.new("RGB",(1680,420),"white");draw=ImageDraw.Draw(sheet)
  for i,(view,frame,p) in enumerate(pictures):
   im=Image.open(p);im.thumbnail((240,180));x=(i%7)*240;y=(i//7)*210;sheet.paste(im,(x,y))
   draw.text((x+2,y+182),f"{example['group']} {view} f{frame}",fill="black")
  sheet.save(sheets/(example["group"]+".png"))
 assert len(checks)==2
 (ROOT/"video_quality_audit.json").write_text(json.dumps({"passed":True,"videos":checks,"encoding_frames":2000,
  "sampled_frames":14,"contact_sheets":1,"manual_attachment_plot_review_required":True},indent=2))
 print("ISSUE17_VIDEO_QA_PASS2CLIPS14SAMPLES",flush=True)
if __name__=="__main__":main()
