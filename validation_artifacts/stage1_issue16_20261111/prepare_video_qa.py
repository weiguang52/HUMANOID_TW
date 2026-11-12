from pathlib import Path
s=Path('scripts/locomotion/audit_issue15_videos.py').read_text().replace('issue15','issue16')
s=s.replace('assert len(plan)==2','assert len(plan)==1')
s=s.replace('json.loads((ROOT/"reconstruction_state.json").read_text())[example["group"]]','json.loads((ROOT/"reconstruction_state.json").read_text())')
s=s.replace('["front","side"]','["front","side","overhead"]')
s=s.replace('(1680,420)','(1680,630)').replace('assert len(checks)==4','assert len(checks)==3').replace('"encoding_frames":4000','"encoding_frames":3000').replace('"sampled_frames":28','"sampled_frames":21').replace('"contact_sheets":2','"contact_sheets":1').replace('ISSUE15_VIDEO_QA_PASS4CLIPS28SAMPLES','ISSUE16_VIDEO_QA_PASS3CLIPS21SAMPLES')
p=Path('scripts/locomotion/audit_issue16_videos.py');compile(s,str(p),'exec');p.write_text(s)
p=Path('scripts/locomotion/replay_issue16.py');s=p.read_text().replace('EXACT RECORDED1024-ENV TRIAL | NO PHYSICS RE-SIMULATION','ORIGINAL POLICY | EXACT1024-ENV SOURCE | NO PHYSICS RE-SIMULATION')
old='   if i==0:\n    for _ in range(60):\n     if frame.max()>0:break\n     base.sim.render();frame=base.render()'
new='   if i==0:\n    # The first base.render creates its render product; warm that product too.\n    for _ in range(60):\n     base.sim.render();frame=base.render()'
assert old in s;s=s.replace(old,new)
s=s.replace('"source_model":selection["model"]','"source_policy_mode":"original","initial_camera_product_warmup_renders":60,"source_model":selection["model"]')
compile(s,str(p),'exec');p.write_text(s)
p=Path('scripts/locomotion/reconstruct_issue16.py');s=p.read_text().replace("  q=z['values'][:,3:7].astype('float64')","  reset_frames=np.flatnonzero(~z['flags'][:,0].astype(bool)).tolist()\n  q=z['values'][:,3:7].astype('float64')")
s=s.replace("reset_frames=np.flatnonzero(~np.load(trajectory)['flags'][:,0].astype(bool)).tolist()","reset_frames=reset_frames")
compile(s,str(p),'exec');p.write_text(s)
print('THREE_VIEW_VIDEO_QA_AND_RENDER_PRODUCT_WARMUP_READY')
