#!/usr/bin/env python3
"""Side by side source skeleton and URDF kinematic replay; no dynamics."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco,trimesh,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from retarget_humanml3d import DEFAULT_TRAINING_URDF
ROOT=Path('/root/gpufree-data/datasets/practice9/tw75_dataset_v1')
EDGES=[(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]
def models(output):
    tree=ET.parse(DEFAULT_TRAINING_URDF);root=tree.getroot()
    extension=ET.SubElement(root,'mujoco');ET.SubElement(extension,'compiler',discardvisual='false',fusestatic='false')
    meshes=output/'meshes';meshes.mkdir(exist_ok=True)
    for node in root.findall('.//mesh'):
        source=Path(node.get('filename'));dest=meshes/(source.stem+'.obj')
        if not dest.exists():trimesh.load(str(source),force='mesh').export(str(dest),file_type='obj')
        node.set('filename',str(dest))
    urdf=output/'render.urdf';tree.write(urdf)
    model=mujoco.MjModel.from_xml_path(str(urdf));mjcf=output/'render.xml';mujoco.mj_saveLastXML(str(mjcf),model)
    tree=ET.parse(mjcf);root=tree.getroot();world=root.find('worldbody')
    body=ET.Element('body',name='replay_root');ET.SubElement(body,'freejoint',name='replay_free')
    for node in list(world):world.remove(node);body.append(node)
    world.append(body)
    ET.SubElement(world,'geom',name='floor',type='plane',size='200 200 .01',rgba='.88 .89 .91 1',contype='0',conaffinity='0')
    ET.SubElement(world,'light',pos='0 -2 4',dir='0 0 -1',diffuse='.8 .8 .8')
    for geom in body.findall('.//geom'):
        if geom.get('type')=='mesh':geom.set('rgba','.25 .52 .75 1');geom.set('group','1')
        else:geom.set('group','3')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='640',offheight='640')
    tree.write(mjcf)
    robot=mujoco.MjModel.from_xml_path(str(mjcf))
    human=mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="640" offheight="640"/></visual><worldbody><light pos="0 -3 6"/><geom type="plane" size="200 200 .01" rgba=".88 .89 .91 1"/></worldbody></mujoco>')
    return robot,human

def camera(points):
    low=points.reshape(-1,3).min(0);high=points.reshape(-1,3).max(0)
    cam=mujoco.MjvCamera();cam.lookat[:]=(low+high)/2
    cam.distance=max(float(np.linalg.norm(high-low))*1.4,.6);cam.azimuth=135;cam.elevation=-12
    return cam
def run(output,ids):
    output.mkdir(parents=True,exist_ok=True);robot,human=models(output)
    rd=mujoco.MjData(robot);hd=mujoco.MjData(human)
    rr=mujoco.Renderer(robot,640,640);hr=mujoco.Renderer(human,640,640)
    option=mujoco.MjvOption();option.geomgroup[3]=0
    records={x['id']:x for x in json.loads((ROOT/'pilot_training_geometry_aligned/retarget_manifest.json').read_text())['motions']}
    selection={x['id']:x for x in json.loads((ROOT/'selection.json').read_text())}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    report=[]
    for mid in ids:
        record=records[mid];z=np.load(record['file']);q=z['joint_pos'];rp=z['root_pos'];rq=z['root_quat_xyzw']
        names=z['joint_names'].tolist();source=np.load(record['source_file'])[:,:22,[2,0,1]].astype(float)
        source[:,:,:2]-=source[0,0,:2];rp=rp.copy();rp[:,:2]-=rp[0,:2]
        indices=[]
        for n in names:
            j=mujoco.mj_name2id(robot,mujoco.mjtObj.mjOBJ_JOINT,str(n))
            if j<0:raise ValueError('Missing joint '+str(n))
            indices.append(robot.jnt_qposadr[j])
        rc_points=np.concatenate([rp+np.array([0,0,-.29]),rp+np.array([0,0,.2])])
        hc=camera(source);rc=camera(rc_points)
        scale=float(record['quality']['time_scale_applied']);frames=list(range(0,len(q),2))
        path=output/f'{mid}_comparison.mp4';writer=imageio.get_writer(str(path),fps=25,codec='libx264',quality=8,macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'])
        try:
            for oi,i in enumerate(frames):
                phase=i/(len(q)-1);si=phase*(len(source)-1);a=int(si);b=min(a+1,len(source)-1);points=(1-(si-a))*source[a]+(si-a)*source[b]
                mujoco.mj_forward(human,hd);hr.update_scene(hd,camera=hc)
                for a,b in EDGES:
                    geom=hr.scene.geoms[hr.scene.ngeom];hr.scene.ngeom+=1
                    color=np.array([.18,.48,.83,1]) if a in [1,4,7,13,16,18] else np.array([.88,.42,.2,1])
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),color)
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.018,points[a],points[b])
                left=hr.render().copy()
                rd.qpos[:3]=rp[i];rd.qpos[3:7]=rq[i][[3,0,1,2]];rd.qpos[indices]=q[i]
                mujoco.mj_forward(robot,rd);rr.update_scene(rd,camera=rc,scene_option=option);right=rr.render().copy()
                canvas=Image.new('RGB',(1280,736),'#172334');canvas.paste(Image.fromarray(left),(0,60));canvas.paste(Image.fromarray(right),(640,60));draw=ImageDraw.Draw(canvas)
                draw.text((16,8),f'{mid} | HumanML3D source NPY',font=font,fill='white')
                draw.text((656,8),'Corrected URDF | KINEMATIC reference replay',font=font,fill='white')
                draw.text((16,34),f'Phase {phase:5.1%} | source {si/20:.2f}s | slowdown {scale:.2f}x',font=small,fill='#d3dce5')
                draw.text((656,34),'Fixed camera | no policy / no contact dynamics',font=small,fill='#d3dce5')
                draw.text((16,710),selection[mid]['captions'][0][:145],font=small,fill='white')
                writer.append_data(np.asarray(canvas))
                if oi==len(frames)//2:canvas.save(output/f'{mid}_preview.png')
        finally:writer.close()
        report.append({'id':mid,'source':record['source_file'],'reference':record['file'],'urdf':str(DEFAULT_TRAINING_URDF),'caption':selection[mid]['captions'][0],'category':selection[mid]['category'],'quality_pass':record['quality_pass'],'frames':len(frames),'fps':25,'time_scale':scale,'video':str(path),'rendering':'kinematic replay, ground visual only; no contact dynamics','camera':'fixed per clip; same azimuth/elevation; separately fitted scales'})
        (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(mid,'completed',len(frames),flush=True)
    rr.close();hr.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--ids',nargs='+',default=['011921','000005','000009','000308','007981']);a=p.parse_args();run(a.output,a.ids)
