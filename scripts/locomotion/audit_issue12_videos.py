"""Encoding and sampled color/framing audit; manual attachment review is separate."""
import json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
from scipy.ndimage import binary_opening,label,find_objects
from issue12_common import ROOT

def main():
    assert (ROOT/"videos_complete").exists()
    plan=json.loads((ROOT/"video_plan.json").read_text());assert len(plan)==10
    frames=ROOT/"video_review_frames";frames.mkdir(exist_ok=True)
    contacts=ROOT/"video_contact_sheets";contacts.mkdir(exist_ok=True)
    checks={};sample_frames=[0,250,500,750,999]
    for entry in plan:
        pictures=[]
        for view in ["front","side"]:
            p=Path(entry["views"][view]["output"])
            d=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
                "-show_entries","stream=codec_name,width,height,nb_frames,r_frame_rate,duration",
                "-of","json",str(p)]))["streams"][0]
            assert d["codec_name"]=="h264" and int(d["nb_frames"])==1000 and d["r_frame_rate"]=="50/1"
            assert (d["width"],d["height"])==(960,720)
            samples=[]
            for frame in sample_frames:
                out=frames/(p.stem+f"_frame{frame}.png")
                subprocess.run(["ffmpeg","-nostdin","-v","error","-threads","1","-y","-i",str(p),
                    "-vf",f"select=eq(n\\,{frame})","-frames:v","1",str(out)],check=True)
                im=Image.open(out).convert("RGB");arr=np.asarray(im).astype("int16")
                assert arr.max()>0
                mask=(arr.max(-1)-arr.min(-1)>35)&(arr.max(-1)>70);mask[:42]=False
                mask=binary_opening(mask,structure=np.ones((3,3)))
                labels,_=label(mask)
                for component,region in enumerate(find_objects(labels),1):
                    h=region[0].stop-region[0].start;w=region[1].stop-region[1].start
                    if h<=5 and w>=20:
                        patch=mask[region];patch[labels[region]==component]=False
                yy,xx=np.where(mask);assert len(xx)>100,(p,frame,"missing colored robot")
                bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
                assert bbox[0]>10 and bbox[1]>42 and bbox[2]<950 and bbox[3]<710,(p,frame,bbox)
                samples.append({"frame":frame,"bbox":bbox})
                pictures.append((view,frame,out))
            checks[p.name]={"probe":d,"samples":samples,"color_fullbody_black_frame_checks":True,
                           "scope":"illustrative independent one-env rollout, not acceptance sample"}
        sheet=Image.new("RGB",(1200,420),"white");draw=ImageDraw.Draw(sheet)
        for i,(view,frame,path) in enumerate(pictures):
            im=Image.open(path);im.thumbnail((240,180));x=(i%5)*240;y=(i//5)*210
            sheet.paste(im,(x,y));draw.text((x+2,y+182),entry["model"]+" "+view+f" f{frame}",fill="black")
        sheet.save(contacts/(entry["model"]+".png"))
    assert len(checks)==20
    result={"passed":True,"videos":checks,"sampled_frames":100,"all_frames_encoding_counted":20000,
            "manual_attachment_and_plot_review_required_separately":True}
    (ROOT/"video_quality_audit.json").write_text(json.dumps(result,indent=2))
    print("ISSUE12_VIDEO_QA_PASS",len(checks),"clips/100sampled frames",flush=True)
if __name__=="__main__":main()
