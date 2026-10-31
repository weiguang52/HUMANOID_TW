"""Render recorded physics states, with no continuation after termination."""
import argparse,json,xml.etree.ElementTree as E
from pathlib import Path
import numpy as np,mujoco,mediapy
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--mjcf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
tree=E.fromstring(a.mjcf.read_text());world=tree.find('worldbody');E.SubElement(world,'light',pos='0 0 2',directional='true');E.SubElement(world,'geom',name='replay_floor',type='plane',size='0 0 .01',rgba='.75 .78 .82 1');m=mujoco.MjModel.from_xml_string(E.tostring(tree,encoding='unicode'));m.vis.global_.offwidth=960;m.vis.global_.offheight=640;d=mujoco.MjData(m);mapping=json.loads((a.collection/'control_mapping.json').read_text());adr=[m.joint(n).qposadr[0] for n in mapping['state_joint_names']];records=[]
with mujoco.Renderer(m,height=640,width=960) as renderer:
 for row in json.loads((a.collection/'results.json').read_text())['rows']:
  # Render every fixed episode, including failures.
  states=np.load(a.collection/row['trace'])['next_state'];paths={v:a.output/(Path(row['trace']).stem+'_'+v+'.mp4') for v in ['front','side']}
  with mediapy.VideoWriter(str(paths['front']),(640,960),fps=50) as f,mediapy.VideoWriter(str(paths['side']),(640,960),fps=50) as s:
   for i,state in enumerate(states):
    d.qpos[:3]=state[:3];d.qpos[3:7]=state[3:7];d.qpos[adr]=state[13:43];mujoco.mj_forward(m,d);cam=mujoco.MjvCamera();cam.distance=1.0;cam.elevation=-10;cam.lookat[:]=[state[0],state[1],.28]
    for view,angle,writer in [('front',180,f),('side',90,s)]:
     cam.azimuth=angle;renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((10,10),'RECORDED PHYSICS / OPEN LOOP PD / '+row['mode']+' / '+str(round((i+1)*.02,2))+'s',fill='white');writer.add_image(np.asarray(im))
  records.append(dict(id=row['id'],mode=row['mode'],frames=len(states),videos={k:v.name for k,v in paths.items()},termination_reasons=row['termination_reasons']));print(row['trace'],len(states),flush=True)
(a.output/'replay_index.json').write_text(json.dumps(records,indent=2))
