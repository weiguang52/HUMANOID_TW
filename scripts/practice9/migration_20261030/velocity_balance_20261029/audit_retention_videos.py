from pathlib import Path
import sys,json,subprocess,numpy as np
from PIL import Image,ImageDraw
state=Path(sys.argv[1]);videos=state/'videos';rows=[]
checkpoint=Path((state/'checkpoint').read_text().strip());training_contract=json.loads((checkpoint.parent/'params/training_speed_blend.json').read_text())
for view in ['front','side']:
 d=videos/'retention'/view;jobs=json.loads((d/'jobs.json').read_text());assert Path(str(d/'jobs.json')+'.done').read_text().strip()=='completed';log=(d/'run.log').read_text()
 for j in jobs:
  p=Path(j['output']);assert 'BATCH_COMPLETED '+str(p) in log
  ref=state/'retention'/p.name;meta=json.loads(p.read_text());gold=json.loads(ref.read_text());a=np.load(p.with_suffix('.npz'));b=np.load(ref.with_suffix('.npz'))
  parity={k:bool(np.array_equal(a[k],b[k])) for k in ['q','action','anchor_actual','reference_q','terminal']}
  for k in ['control_runtime','policy_blend','action_filter','checkpoint']:parity[k]=meta[k]==gold[k]
  parity['training_speed_blend_contract']=meta['training_speed_blend_contract']['contract']==training_contract==gold['training_speed_blend_contract']['contract']
  expected_warmup=j.get('video_warmup_frames',0)
  warmup=meta['video_camera'].get('render_warmup',{})
  parity['render_initialization_contract']=warmup.get('frames',0)==expected_warmup and (not expected_warmup or (warmup['physics_and_command_unchanged'] and warmup['RNG_unchanged']))
  x=np.load(p.with_suffix('.blend.npz'));y=np.load(ref.with_suffix('.blend.npz'));parity['blend_trace']=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
  assert np.all(x['speed_and_candidate_weight'][...,1]==0) and len(a['terminal'])==1000 and not a['terminal'].any()
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate,width,height','-of','json',str(p.with_suffix('.mp4'))]))['streams'][0]
  camera=np.load(p.with_suffix('.camera.npz'));expected=500
  good=all(parity.values()) and meta['video_camera']['view']==view and info['codec_name']=='h264' and info['r_frame_rate']=='25/1' and info['width']==1280 and info['height']==720 and int(info['nb_frames'])==expected==len(camera['reset'])
  assert good,(str(p),parity,info)
  rows.append({'id':p.stem,'view':view,'video':str(p.with_suffix('.mp4')),'passed':bool(good),'parity':parity,'encoding':info,'source_type':'modified synthetic diagnostic','source_template_motion':'000016','source_template_is_not_time_aligned_counterpart':True,'teacher_fallback_all_steps':True})
assert len(rows)==4
out={'complete':True,'expected':4,'observed':4,'all_observed_pass':all(x['passed'] for x in rows),'rows':rows,'visual_reviewed':False,'scope':'Encoding and numeric replay parity only; synthetic references constructed from000016, not original HumanML3D motions or original contact truth.'}
(videos/'retention_video_audit.json').write_text(json.dumps(out,indent=2))
qa=videos/'synthetic_video_qa';qa.mkdir(exist_ok=True);sheet=Image.new('RGB',(1280,260*len(rows)),(245,245,245));draw=ImageDraw.Draw(sheet);frames=[]
for row,item in enumerate(rows):
 for col,t in enumerate([2.,10.,18.]):
  frame=qa/(item['id']+'.'+item['view']+'.'+str(col)+'.jpg')
  subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(t),'-i',item['video'],'-frames:v','1','-vf','scale=426:240','-y',str(frame)],check=True)
  with Image.open(frame) as image:sheet.paste(image,(426*col,260*row+20))
  draw.text((426*col+3,260*row+3),item['id']+' '+item['view']+' '+str(t)+'s synthetic teacher fallback',fill=(0,0,0));frames.append(str(frame))
path=qa/'retention.all.jpg';sheet.save(path)
(qa/'audit.json').write_text(json.dumps({'sheet':str(path),'frames':frames,'visual_reviewed':False,'scope':'12 sparse frames; does not prove continuous visual quality'},indent=2))
main=json.loads((videos/'video_audit.json').read_text());source=json.loads((videos/'source_video_audit.json').read_text())
assert main['complete'] and main['observed']==42 and main['all_observed_pass'] and source['complete'] and len(source['rows'])==21
(videos/'delivery_video_audit.json').write_text(json.dumps({'complete':True,'robot_videos':46,'main_motion_videos':42,'synthetic_retention_videos':4,'original_human_videos':21,'all_encoding_and_replay_checks_pass':True,'visual_reviewed':False,'final_acceptance':False,'scope':'Artifact encoding/parity completion only; actual visual inspection and full quality decision still required'},indent=2))
print('Passed main42 + synthetic4 replay videos, 21 original source videos; visual review pending')
