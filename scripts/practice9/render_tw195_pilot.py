"""Fixed-camera kinematic URDF replay; no physics or policy claim."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import pathlib,json,numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
from render_tw75_pairs import models,camera
S=pathlib.Path('/root/gpufree-data/datasets/practice9/tw195_v1');O=pathlib.Path('validation_artifacts/tw195_setup');O.mkdir(parents=True,exist_ok=True)
robot,human=models(S);rd=mujoco.MjData(robot);renderer=mujoco.Renderer(robot,640,640);option=mujoco.MjvOption();option.geomgroup[3]=0
m=json.load(open('/root/gpufree-data/datasets/practice9/tw154_v1/validation/manifest.json'))['motions']
for mid in ['000016','000832','000211']:
 old=np.load(next(r['file'] for r in m if r['id']==mid));new=np.load(S/'pilot_smooth'/f'{mid}.npz');indices=[robot.jnt_qposadr[mujoco.mj_name2id(robot,mujoco.mjtObj.mjOBJ_JOINT,str(n))] for n in old['joint_names']];rp=old['root_pos'];cam=camera(np.concatenate([rp+[0,0,-.29],rp+[0,0,.2]]));cam.azimuth=90
 writer=imageio.get_writer(str(O/f'{mid}_before_after.mp4'),fps=25,codec='libx264',macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p'])
 for t in range(0,len(rp),2):
  canvas=Image.new('RGB',(1280,688),'#172334');draw=ImageDraw.Draw(canvas)
  for j,(a,label) in enumerate([(old,'Original reference'),(new,'Contact IK reference')]):
   rd.qpos[:3]=a['root_pos'][t];rd.qpos[3:7]=a['root_quat_wxyz'][t];rd.qpos[indices]=a['joint_pos'][t];mujoco.mj_forward(robot,rd);renderer.update_scene(rd,camera=cam,scene_option=option);canvas.paste(Image.fromarray(renderer.render()),(j*640,48));draw.text((j*640+10,10),f'{mid} {label} | {t/50:.2f}s | KINEMATIC, not dynamics',fill='white')
  writer.append_data(np.asarray(canvas))
  if t==2*(len(rp)//4):canvas.save(O/f'{mid}_preview.png')
 writer.close();print(mid,'done',flush=True)
renderer.close()
