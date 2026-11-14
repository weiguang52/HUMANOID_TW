"""Paired phase-factor saved-state playback with common world-XY trail scale."""
import json,math,numpy as np,torch,imageio.v2 as imageio
from pathlib import Path
from PIL import Image,ImageDraw
from issue20_common import sha

def replay(base,args):
 p=Path(args.trajectory_replay);s=json.loads(p.read_text());phase=s['phase'];trajectory=Path(s['trajectory']);assert sha(trajectory)==s['trajectory_sha256'];z=np.load(trajectory);values=z['root_and_joints'];assert values.shape==(1000,37);robot=base.scene['robot'];assert list(robot.joint_names)==s['joint_names']
 yawdata=np.load(s['yaw_file']);yaw=yawdata['heading_yaw'][:,0];start=float(yawdata['heading_start'][0]);delta=np.arctan2(np.sin(yaw-np.r_[start,yaw[:-1]]),np.cos(yaw-np.r_[start,yaw[:-1]]));failed=int(yawdata['first_failure_step'][0]);valid=np.ones(1000,bool)
 if failed>=0:valid[failed:]=False
 drift=np.cumsum(np.where(valid,delta,0))*180/np.pi;xy=values[:,:2];center=np.asarray(s['common_xy_center']);span=s['common_xy_span'];mapped=np.stack([845+(xy[:,0]-center[0])*90/span,161-(xy[:,1]-center[1])*90/span],1)
 output=Path(args.output);path=output.parent/'videos'/(output.stem+'.mp4');path.parent.mkdir(parents=True,exist_ok=True);counter=base.common_step_counter
 with imageio.get_writer(str(path),fps=50) as writer:
  for i,row in enumerate(values):
   pose=torch.as_tensor(row[:7],device=base.device).unsqueeze(0).clone();pose[:,:3]+=base.scene.env_origins[:1];joints=torch.as_tensor(row[7:],device=base.device).unsqueeze(0);robot.write_root_pose_to_sim(pose);robot.write_root_velocity_to_sim(torch.zeros(1,6,device=base.device));robot.write_joint_state_to_sim(joints,torch.zeros_like(joints));base.sim.forward()
   pos=pose[0,:3].cpu().tolist();angle=float(yaw[i])+(math.pi/2 if args.camera_view=='side' else 0);eye=(pos[0]+.25,pos[1]+.10,pos[2]+1.6) if args.camera_view=='overhead' else (pos[0]+1.2*math.cos(angle),pos[1]+1.2*math.sin(angle),pos[2]+.35);base.sim.set_camera_view(eye=eye,target=(pos[0],pos[1],pos[2]+.05))
   for _ in range(60 if i==0 else 2):base.sim.render()
   frame=base.render()
   if i==0:
    for _ in range(60):base.sim.render();frame=base.render()
   assert frame.max()>0;im=Image.fromarray(frame[:,:,:3]);draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,78),fill=(20,20,20));draw.text((10,4),f'ISSUE20 | {s["usd"].upper()} USD | START PHASE k={phase} | {args.camera_view.upper()}',fill='white');draw.text((10,23),f'Train seed {s["training_seed"]}, eval seed 1, fixed env 0 | forward 0.5 m/s | t={(i+1)*.02:.2f}s',fill='white');draw.text((10,43),f'Exact saved states, NO dynamics integration | initial-prefix drift {drift[i]:+.2f} deg'+(' | AFTER FIRST RESET' if not valid[i] else ''),fill='white');draw.text((10,62),f'Seed chosen by maximum |mean D|={abs(s["seed_mean_D_degrees"]):.2f}deg; env0 fixed, not maximum trial',fill='white')
   draw.rectangle((740,78,952,242),fill=(24,24,24));draw.text((747,82),'WORLD XY + HEADING',fill='white');points=[tuple(v) for v in mapped[:min(i+1,failed if failed>=0 else 1000)]]
   if len(points)>1:draw.line(points,fill=(80,230,180),width=2)
   px,py=mapped[i];draw.ellipse((px-3,py-3,px+3,py+3),fill=(255,230,80));draw.line((px,py,px+14*math.cos(float(yaw[i])),py-14*math.sin(float(yaw[i]))),fill=(255,90,80),width=3);draw.text((747,224),f'{span:.2f}m | X right,Y up',fill='white');draw.rectangle((0,680,960,720),fill=(20,20,20));draw.text((15,690),'Fixed world orientation overhead; front/side follow translation and heading. Paired source frame/time.',fill='white');writer.append_data(np.asarray(im))
 assert base.common_step_counter==counter
 output.write_text(json.dumps({'passed':True,'phase':phase,'usd':s['usd'],'source_trajectory_sha256':s['trajectory_sha256'],'source_selection_sha256':sha(p),'replay_source_sha256':sha(__file__),'render_adapter_sha256':sha(Path(__file__).with_name('render_eval_issue20.py')),'frames':1000,'fps':50,'seconds':20,'physics_integrated':False,'view':args.camera_view,'camera_follows_translation':True,'camera_follows_heading':args.camera_view!='overhead','video':str(path),'video_sha256':sha(path),'humanml_source_npy':'N/A: stage1 velocity command, not motion retargeting'},indent=2)+'\n')
