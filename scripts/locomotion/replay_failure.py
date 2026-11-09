"""Render exact recorded root/joint states; never integrate a new trajectory."""
import json,math,hashlib
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw

def replay(base,args):
    selection=json.loads(Path(args.trajectory_replay).read_text())
    meta=json.loads(Path(selection['source_metadata']).read_text())
    z=np.load(selection['source_npz']);index=selection['record_index']
    values=z['values'][index][z['valid'][index]]
    robot=base.scene['robot'];assert list(robot.joint_names)==meta['joint_names']
    fields=meta['fields']
    def part(row,name):return row[slice(*fields[name])]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    path=output.parent/'videos'/(output.stem+'.mp4');path.parent.mkdir(exist_ok=True)
    initial_counter=base.common_step_counter
    with imageio.get_writer(str(path),fps=50) as writer:
        for i,row in enumerate(values):
            pose=torch.as_tensor(part(row,'root_pose_local'),device=base.device).unsqueeze(0).clone()
            pose[:,:3]+=base.scene.env_origins[:1]
            joints=torch.as_tensor(part(row,'joint_pos'),device=base.device).unsqueeze(0)
            velocity=torch.as_tensor(part(row,'root_velocity_world'),device=base.device).unsqueeze(0)
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(velocity)
            robot.write_joint_state_to_sim(joints,torch.zeros_like(joints))
            base.sim.forward()
            q=pose[0,3:].cpu().tolist();pos=pose[0,:3].cpu().tolist()
            yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
            angle=yaw if args.camera_view=='front' else yaw+math.pi/2
            eye=(pos[0]+1.2*math.cos(angle),pos[1]+1.2*math.sin(angle),pos[2]+.35)
            base.sim.set_camera_view(eye=eye,target=(pos[0],pos[1],pos[2]+.05))
            for _ in range(60 if i==0 else 2):base.sim.render()
            frame=base.render()
            if i==0:
                for _ in range(60):
                    if frame.max()>0:break
                    base.sim.render();frame=base.render()
            assert frame.max()>0
            image=Image.fromarray(frame[:,:,:3]);draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,image.width,40),fill=(20,20,20))
            draw.text((10,5),'RECORDED PRE-RESET STATE REPLAY | NO PHYSICS RE-SIMULATION',fill='white')
            relative=(i-len(values)+1)*meta['step_dt']
            draw.text((10,22),f"{selection['mode']} | {args.camera_view} | t={relative:+.2f}s",fill='white')
            writer.append_data(np.asarray(image))
    assert base.common_step_counter==initial_counter
    output.write_text(json.dumps({'source_selection':str(args.trajectory_replay),'source_npz_sha256':hashlib.sha256(Path(selection['source_npz']).read_bytes()).hexdigest(),'source_model':selection['model'],'source_environment':selection['env_id'],'source_command':selection['command'],'view':args.camera_view,'frames':len(values),'fps':50,'physics_integrated':False,'first_frame_warmup_renders':60,'root_and_joint_states':'exact recorded terminal pre-reset window','joint_velocity_for_display':'zero; no physics stepping','camera':'follows translation and heading','video':str(path)},indent=2))
