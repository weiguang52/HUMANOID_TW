"""Colored kinematic asset QA; explicitly not learned policy or physics success."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--mjcf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=1000)
    a = p.parse_args()
    assert a.max_steps > 0
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output / 'audit.json').exists(), 'Refusing completed output overwrite'
    manifest = json.loads(a.manifest.read_text())
    assert manifest['joint_coordinate_contract'] == 'tw123_yaw_waist_v3'
    tree = ET.fromstring(a.mjcf.read_text())
    world = tree.find('worldbody')
    ET.SubElement(world, 'light', pos='0 0 2', dir='0 0 -1', directional='true')
    ET.SubElement(world, 'geom', name='reference_floor', type='plane', size='0 0 .01', rgba='.75 .78 .82 1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
    assert model.nq == 37 and model.nv == 36
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 640
    model.vis.global_.fovy = 45
    data = mujoco.MjData(model)
    font = ImageFont.load_default(size=22)
    records = []
    with mujoco.Renderer(model, height=640, width=960) as renderer:
        for row in manifest['motions']:
            with np.load(row['file']) as motion:
                fps = float(motion['fps'][0])
                assert fps == 50.0
                joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(n)) for n in motion['joint_names']]
                body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(n)) for n in motion['body_names']]
                assert min(joint_ids) >= 0 and min(body_ids) > 0
                qadr = model.jnt_qposadr[joint_ids]
                steps = min(a.max_steps, len(motion['joint_pos']) - 1)
                q = motion['root_quat_wxyz'][0]
                yaw = math.degrees(math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.distance = 1.15
                cam.elevation = -10
                paths = {v: a.output / f"{row['id']}_reference_{v}.mp4" for v in ('front', 'side')}
                max_fk_error = 0.0
                with mediapy.VideoWriter(str(paths['front']), (640, 960), fps=50) as front, mediapy.VideoWriter(str(paths['side']), (640, 960), fps=50) as side:
                    for frame in range(1, steps + 1):
                        data.qpos[:3] = motion['root_pos'][frame]
                        data.qpos[3:7] = motion['root_quat_wxyz'][frame]
                        data.qpos[qadr] = motion['joint_pos'][frame]
                        mujoco.mj_forward(model, data)
                        error = np.linalg.norm(data.xpos[body_ids] - motion['body_pos_w'][frame], axis=-1).max()
                        max_fk_error = max(max_fk_error, float(error))
                        assert np.isfinite(error) and error < 2e-6, error
                        root = data.qpos[:3]
                        cam.lookat[:] = [root[0], root[1], max(.22, root[2]+.03)]
                        for view, offset, writer in [('front', 180, front), ('side', 90, side)]:
                            cam.azimuth = yaw + offset
                            renderer.update_scene(data, camera=cam)
                            image = Image.fromarray(renderer.render())
                            draw = ImageDraw.Draw(image)
                            draw.rectangle((0, 0, 960, 34), fill=(15, 15, 15))
                            draw.text((8, 5), f"REFERENCE / NOT POLICY | {row['id']} | {view} | {frame/50:.2f}s", font=font, fill='white')
                            writer.add_image(np.asarray(image))
                record = dict(id=row['id'], steps=steps, fps=50, duration_s=steps/50, max_body_fk_error_m=max_fk_error, motion_sha256=hashlib.sha256(Path(row['file']).read_bytes()).hexdigest(), videos={v:path.name for v,path in paths.items()})
                records.append(record)
                print(json.dumps(record), flush=True)
    report = dict(kind='kinematic_reference_not_policy', mjcf_sha256=hashlib.sha256(a.mjcf.read_bytes()).hexdigest(), manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(), rows=records, note='Direct qpos reference animation without dynamics or policy. Original robot meshes and colors; only presentation floor/light added. Does not demonstrate balance, tracking or learned success. Cameras retain initial heading during turns.')
    (a.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    source_index = a.manifest.parent / 'source_videos/index.json'
    source_rows = {r['id']: r for r in json.loads(source_index.read_text())} if source_index.exists() else {}
    links = []
    for r in records:
        video_links = [f'<a href="{html.escape(path)}">{view}</a>' for view, path in r['videos'].items()]
        if r['id'] in source_rows:
            original = source_index.parent / source_rows[r['id']]['video']
            assert original.is_file()
            video_links.append(f'<a href="{html.escape(os.path.relpath(original, a.output))}">original HumanML3D</a>')
        links.append(f"<tr><td>{html.escape(r['id'])}</td><td>{r['duration_s']:.2f}s</td><td>"+' | '.join(video_links)+'</td></tr>')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TW kinematic reference</title><h1>REFERENCE / NOT POLICY</h1><p>Direct reference animation; no physics integration or learned control. Does not demonstrate balance or tracking success. See policy evaluation separately. Robot 50fps and original HumanML3D 20fps are not synchronized; robot references may be time-stretched.</p><table>'+''.join(links)+'</table><p><a href="audit.json">Provenance / FK audit</a></p>')



if __name__ == '__main__':
    main()
