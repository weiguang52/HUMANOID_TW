from pathlib import Path
p=Path('scripts/locomotion/reconstruct_issue16.py');s=p.read_text()
s=s.replace("recorded=z['heading_yaw'][:,selected['env_id']].astype('float64')","recorded=z['heading_yaw'][:,selected['env_id']].astype('float64')\n  initial_yaw=float(z['heading_start'][selected['env_id']])")
s=s.replace("state=dict(selected,trajectory=str(trajectory),trajectory_metadata=str(metadata)","state=dict(selected,heading_start_yaw=initial_yaw,reset_frames=np.flatnonzero(~np.load(trajectory)['flags'][:,0].astype(bool)).tolist(),trajectory=str(trajectory),trajectory_meta=str(metadata),trajectory_metadata=str(metadata)")
compile(s,str(p),'exec');p.write_text(s)
old=Path('scripts/locomotion/replay_issue15.py').read_text();s=old.replace('issue15','issue16').replace('forward0.3','forward0.5').replace('"forward_03"','"forward_05"')
before='   angle=yaw if args.camera_view=="front" else yaw+math.pi/2\n   base.sim.set_camera_view(eye=(pos[0]+1.2*math.cos(angle),pos[1]+1.2*math.sin(angle),pos[2]+.35),target=(pos[0],pos[1],pos[2]+.05))'
after='   angle=yaw if args.camera_view=="front" else yaw+math.pi/2\n   eye=(pos[0]+.25,pos[1]+.10,pos[2]+1.6) if args.camera_view=="overhead" else (pos[0]+1.2*math.cos(angle),pos[1]+1.2*math.sin(angle),pos[2]+.35)\n   base.sim.set_camera_view(eye=eye,target=(pos[0],pos[1],pos[2]+.05))'
assert before in s;s=s.replace(before,after)
s=s.replace('MEDIAN ILLUSTRATION','MEDIAN ABS-HEADING TRIAL')
s=s.replace('"camera":"follows translation and heading"','"camera":"follows translation; fixed world heading, near overhead" if args.camera_view=="overhead" else "follows translation and heading","camera_follows_heading":args.camera_view!="overhead","camera_follows_translation":True')
s=s.replace('"exact same post-step states and times for paired front/side views; FT5/control examples are not synchronized"','"Exact same source states/times across front, side and overhead; selected by pooled original20s absolute heading drift, without survival filtering"')
p=Path('scripts/locomotion/replay_issue16.py');compile(s,str(p),'exec');p.write_text(s)
s=Path('scripts/locomotion/render_eval_issue15.py').read_text().replace('issue15','issue16').replace('0ec49f358d1634b62b7f8b149381809b54d23b9b','770014ee9a4393da6dff34eced89a04d83e788dd')
needle='sys.argv[0]=str(p)'
s=s.replace(needle,"assert source.count('choices=[\"front\",\"side\"]')==1\nsource=source.replace('choices=[\"front\",\"side\"]','choices=[\"front\",\"side\",\"overhead\"]')\n"+needle)
p=Path('scripts/locomotion/render_eval_issue16.py');compile(s,str(p),'exec');p.write_text(s)
print('ISSUE16_RECONSTRUCTION_BASELINE_YAW_AND_THREE_VIEW_REPLAY_READY')
