import sys,pathlib,tempfile,types,importlib.util
import torch,numpy as np
from unittest.mock import patch
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'scripts/rsl_rl'))
from batch_video import BatchVideo
class Fake:
    def __init__(self):
        self.unwrapped=self;self.device='cpu';self.calls=0
        self._sim_step_counter=2;self.common_step_counter=0
        self.data=types.SimpleNamespace(root_pos_w=torch.zeros(1,3),root_quat_w=torch.tensor([[1.,0,0,0]]),root_state_w=torch.zeros(1,13),joint_pos=torch.zeros(1,30),joint_vel=torch.zeros(1,30))
        self.scene={'robot':types.SimpleNamespace(data=self.data)}
        command=types.SimpleNamespace(time_steps=torch.zeros(1,dtype=torch.long),motion_ids=torch.zeros(1,dtype=torch.long))
        self.command_manager=types.SimpleNamespace(get_term=lambda _:command)
        self.action_manager=types.SimpleNamespace(action=torch.zeros(1,30),prev_action=torch.zeros(1,30))
        self.sim=types.SimpleNamespace(current_time=.02,render=self.tick,set_camera_view=lambda **kw:None)
    def tick(self):self.calls+=1
    def render(self,recompute=False):
        if not recompute:self.tick()
        a=np.zeros((24,32,3),np.uint8)
        if self.calls>2:a[3:21,8:24]=[180,20,70]
        return a
class Writer:
    def __init__(self,*a,**kw):self.frames=[]
    def append_data(self,image):self.frames.append(image.copy())
    def close(self):pass
oldpath=pathlib.Path('/root/gpufree-data/datasets/practice9/training_speed_blend_20261026/source_before_render_fix/batch_video.py')
spec=importlib.util.spec_from_file_location('old_batch_video',oldpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as d,patch('imageio.v2.get_writer',Writer):
    batches=[]
    for cls in (module.BatchVideo,BatchVideo):
        env=Fake();clip=cls(env,pathlib.Path(d)/(cls.__module__+'.json'),'side')
        assert env.calls==0
        for i in range(10):
            env.data.root_pos_w[0,0]=i*.01;clip.append(i==5)
        clip.close();batches.append(clip)
    a,b=batches
    assert a.audit==b.audit and len(a.trace)==len(b.trace)==10
    assert all(np.array_equal(x,y) for x,y in zip(a.writer.frames,b.writer.frames))
    assert all(np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1]) and x[2]==y[2] for x,y in zip(a.trace,b.trace))
    env=Fake();clip=BatchVideo(env,pathlib.Path(d)/'warm.json','side',warmup_frames=8)
    assert env.calls==8 and len(clip.writer.frames)==len(clip.trace)==0
    clip.append();assert len(clip.writer.frames)==1 and clip.writer.frames[0].max()==180 and len(clip.trace)==1
    assert clip.audit['render_warmup']['physics_and_command_unchanged'];clip.close()
print('PASS exact legacy frames/camera/audit with warmup0; explicit8 pre-renders create no recorded frames or physical steps')
