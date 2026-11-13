"""Exact-state20s replay with push-time annotations; never integrate new dynamics."""
import json,math,hashlib
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw
_REPLAY_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_RENDER_ADAPTER_SHA256=hashlib.sha256(Path(__file__).with_name("render_eval_issue18.py").read_bytes()).hexdigest()
def replay(base,args):
 selection=json.loads(Path(args.trajectory_replay).read_text())
 meta=json.loads(Path(selection["trajectory_meta"]).read_text());z=np.load(selection["trajectory"])
 assert hashlib.sha256(Path(selection["trajectory"]).read_bytes()).hexdigest()==selection["trajectory_sha256"]
 values=z["values"];flags=z["flags"];pushes=z["push_steps"]*.02;robot=base.scene["robot"]
 assert list(robot.joint_names)==meta["joint_names"] and values.shape==(1000,43)
 output=Path(args.output);path=output.parent/"videos"/(output.stem+".mp4");path.parent.mkdir(parents=True,exist_ok=True)
 initial_counter=base.common_step_counter
 q=values[:,3:7]
 angles=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
 previous=np.concatenate([[selection["heading_start_yaw"]],angles[:-1]])
 changes=np.arctan2(np.sin(angles-previous),np.cos(angles-previous))
 drift=np.cumsum(np.where(flags[:,0],changes,0.))*180/np.pi
 xy=values[:,:2];center=(xy.min(0)+xy.max(0))/2
 span=max(float(np.ptp(xy,axis=0).max()),.25)
 mapped=np.stack([845+(xy[:,0]-center[0])*90/span,161-(xy[:,1]-center[1])*90/span],1)
 valid_mask=flags[:,0].astype(bool)
 segment_starts=np.flatnonzero(valid_mask & np.r_[True,~valid_mask[:-1]])
 segment_ends=np.flatnonzero(valid_mask & np.r_[~valid_mask[1:],True])+1

 with imageio.get_writer(str(path),fps=50) as writer:
  for i,row in enumerate(values):
   pose=torch.as_tensor(row[:7],device=base.device).unsqueeze(0).clone();pose[:,:3]+=base.scene.env_origins[:1]
   velocity=torch.as_tensor(row[7:13],device=base.device).unsqueeze(0)
   joints=torch.as_tensor(row[13:],device=base.device).unsqueeze(0)
   robot.write_root_pose_to_sim(pose);robot.write_root_velocity_to_sim(velocity)
   robot.write_joint_state_to_sim(joints,torch.zeros_like(joints));base.sim.forward()
   q=pose[0,3:].cpu().tolist();pos=pose[0,:3].cpu().tolist()
   yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
   angle=yaw if args.camera_view=="front" else yaw+math.pi/2
   eye=(pos[0]+.25,pos[1]+.10,pos[2]+1.6) if args.camera_view=="overhead" else (pos[0]+1.2*math.cos(angle),pos[1]+1.2*math.sin(angle),pos[2]+.35)
   base.sim.set_camera_view(eye=eye,target=(pos[0],pos[1],pos[2]+.05))
   for _ in range(60 if i==0 else 2):base.sim.render()
   frame=base.render()
   if i==0:
    # The first base.render creates its render product; warm that product too.
    for _ in range(60):
     base.sim.render();frame=base.render()
   assert frame.max()>0
   im=Image.fromarray(frame[:,:,:3]);draw=ImageDraw.Draw(im);draw.rectangle((0,0,im.width,78),fill=(20,20,20))
   time=(i+1)*.02;inside=bool(flags[i,2]);valid=bool(flags[i,0])
   status="RESET (excluded metric frame)" if not valid else ("RECOVERY WINDOW" if inside else "OUTSIDE WINDOW")
   draw.text((10,4),"STRICT PROJECTED SYM_INT | INTEGER CLOCK | EXACT SAVED STATES | NO DYNAMICS RE-SIMULATION",fill="white")
   draw.text((10,23),f"{selection['group']} | train seed{selection['training_seed']} eval seed{selection['eval_seed']} env{selection['env_id']} | {args.camera_view} | t={time:.2f}s",fill="white")
   draw.text((10,43),status+(" | AFTER FIRST RESET" if not flags[i,1] else "")+" | push times: "+", ".join(f"{p:.2f}s" for p in pushes),fill=(255,120,80) if inside or not valid else "white")
   draw.rectangle((0,680,im.width,im.height),fill=(20,20,20));left,right=30,930
   draw.line((left,704,right,704),fill=(180,180,180),width=2)
   for p in pushes:
    x=left+(right-left)*p/20;end=left+(right-left)*min(p+1,20)/20
    draw.rectangle((x,697,end,701),fill=(110,50,50));draw.line((x,692,x,711),fill=(255,85,85),width=2)
   for reset in selection["reset_frames"]:
    xr=left+(right-left)*(reset+1)*.02/20;draw.line((xr,692,xr,711),fill=(255,220,70),width=2)
   x=left+(right-left)*time/20;draw.line((x,690,x,714),fill="white",width=2)
   draw.text((30,682),"CLEAN: no pushes | white cursor: elapsed time | yellow ticks: resets",fill="white");draw.text((900,682),"20s",fill="white")
   draw.rectangle((740,78,952,242),fill=(24,24,24))
   draw.text((747,82),"WORLD XY TRAIL + HEADING",fill="white")
   points=[tuple(p) for p in mapped[:i+1]]
   for start,end in zip(segment_starts,segment_ends):
    right=min(i+1,int(end))
    if right>int(start)+1:draw.line(points[int(start):right],fill=(80,230,180),width=2)
   px,py=mapped[i];draw.ellipse((px-3,py-3,px+3,py+3),fill=(255,230,80))
   draw.line((px,py,px+14*math.cos(float(angles[i])),py-14*math.sin(float(angles[i]))),fill=(255,90,80),width=3)
   draw.text((747,224),f"span {span:.2f}m | X right,Y up",fill="white")
   draw.rectangle((0,40,735,63),fill=(20,20,20))
   draw.text((10,43),f"CLEAN forward0.5 | heading {angles[i]*180/math.pi:+.1f}deg | reset-excluded drift {drift[i]:+.1f}deg"+(" | RESET" if not valid else (" | AFTER FIRST RESET" if not flags[i,1] else "")),fill="white")
   scope="ALL-TRIAL PREFIX" if selection["censored_initial_trials"] else "FULL20s ALL INITIAL TRIALS"
   draw.text((10,64),f"MEDIAN {scope} | abs={selection['heading_abs_degrees']:.2f}deg | prefix={selection['initial_prefix_duration_s']:.2f}s | censored {selection['censored_initial_trials']}/15360",fill="white")
   writer.append_data(np.asarray(im))
 assert base.common_step_counter==initial_counter
 output.write_text(json.dumps({"replay_source_sha256":_REPLAY_SOURCE_SHA256,"render_adapter_sha256":_RENDER_ADAPTER_SHA256,"source_selection":str(args.trajectory_replay),"source_trajectory_sha256":selection["trajectory_sha256"],
  "source_policy_mode":"strict_projected_existing_sym_int","source_integer_clock":True,"selection_scope":selection["selection_scope"],"selection_metric":selection["selection_metric"],"initial_camera_product_warmup_renders":60,"source_model":selection["model"],"source_env_id":selection["env_id"],"group":selection["group"],"command":"forward_05",
  "view":args.camera_view,"frames":1000,"fps":50,"seconds":20,"push_seconds":pushes.tolist(),
  "reset_frame_indices":selection["reset_frames"],"physics_integrated":False,"camera":"follows translation; fixed world heading, near overhead" if args.camera_view=="overhead" else "follows translation and heading","camera_follows_heading":args.camera_view!="overhead","camera_follows_translation":True,
  "source_samples":"Exact same strictly projected sym_int source states/times across front and overhead; all15360 initial trials, selection scope and censoring explicitly recorded",
  "display_mass_material":"nominal authored asset; source randomization is documented but not re-integrated",
  "humanml_source_npy":"N/A: Stage1 velocity commands","video":str(path)},indent=2))
