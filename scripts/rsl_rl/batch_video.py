"""Opt-in matched batch videos; fixed initial heading, horizontal follow."""
import math
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from camera_follow import SmoothCameraFollow

class BatchVideo:
    def __init__(self,env,output,view,warmup_frames=0):
        if view not in ('front','side'):raise ValueError('Expected front or side')
        if type(warmup_frames) is not int or not 0<=warmup_frames<=64:raise ValueError('Invalid explicit renderer warm-up')
        self.env=env;self.robot=env.unwrapped.scene['robot'];self.path=Path(output)
        pos=self.robot.data.root_pos_w[0].cpu().numpy();q=self.robot.data.root_quat_w[0].cpu().numpy()
        w,x,y,z=q;yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        angle=yaw if view=='front' else yaw-math.pi/2
        self.eye=np.array([1.5*math.cos(angle),1.5*math.sin(angle),.2]);self.target=np.array([0.,0.,-.03])
        self.follow=SmoothCameraFollow(pos,.04,.5);self.trace=[]
        self.writer=imageio.get_writer(str(self.path.with_suffix('.mp4')),fps=25,codec='libx264',macro_block_size=None)
        self.audit={'view':view,'fps':25,'initial_yaw_rad':yaw,'eye_offset':self.eye.tolist(),'target_offset':self.target.tolist(),'follow':'horizontal translation; fixed height; fixed initial heading','source_synchronization':False}
        self.env.unwrapped.sim.set_camera_view(eye=pos+self.eye,target=pos+self.target)
        if warmup_frames:
            from render_warmup import warm_up_renderer
            try:self.audit['render_warmup']=warm_up_renderer(self.env.unwrapped,warmup_frames)
            except BaseException:
                self.writer.close();raise
    def append(self,reset=False):
        pos=self.robot.data.root_pos_w[0].cpu().numpy();origin=self.follow.update(pos,reset)
        self.env.unwrapped.sim.set_camera_view(eye=origin+self.eye,target=origin+self.target)
        self.writer.append_data(self.env.unwrapped.render())
        self.trace.append((origin.copy(),pos.copy(),bool(reset)))
    def close(self):
        self.writer.close()
        np.savez_compressed(self.path.with_suffix('.camera.npz'),camera_origin=np.array([x[0] for x in self.trace]),robot_origin=np.array([x[1] for x in self.trace]),reset=np.array([x[2] for x in self.trace]))
