from pathlib import Path
import json,numpy as np,subprocess,datetime
b=Path(__file__).parent;rows=[]
for suffix in ('','_seed123'):
 state=b.parent/f'failure_lookback{suffix}_20261028';job=json.load(open(state/'videos/validation/front/jobs42.json'))[0];p=Path(job['output']);log=(p.parent/'run.log').read_text();assert 'BATCH_COMPLETED '+str(p) in log
 reference=state/'evaluation/candidate'/p.name;meta=json.load(open(p));gold=json.load(open(reference))
 a=np.load(p.with_suffix('.npz'));c=np.load(reference.with_suffix('.npz'));checks={k:np.array_equal(a[k],c[k]) for k in ('q','action','anchor_actual','reference_q','terminal')}
 for key in ('policy_blend','action_filter'):checks[key]=meta[key]==gold[key]
 checks['training_speed_blend_contract']=meta['training_speed_blend_contract']['contract']==gold['training_speed_blend_contract']['contract']
 x=np.load(p.with_suffix('.blend.npz'));y=np.load(reference.with_suffix('.blend.npz'));checks['blend_trace']=x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
 assert all(checks.values()),checks
 video=p.with_suffix('.mp4');info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate,width,height','-of','json',str(video)]))['streams'][0]
 expected=(len(a['terminal'])+1)//2;camera=np.load(p.with_suffix('.camera.npz'))
 assert info['codec_name']=='h264' and info['r_frame_rate']=='25/1' and info['width']==1280 and info['height']==720 and int(info['nb_frames'])==expected==len(camera['reset'])
 image=p.with_suffix('.review.jpg');middle=expected/25/2
 subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(middle),'-i',str(video),'-frames:v','1','-q:v','6','-y',str(image)],check=True)
 row={'state':state.name,'case':'000832','view':'front','encoding':info,'numeric_replay_exact':{k:bool(v) for k,v in checks.items()},'middle_frame_seconds':middle,'frame':str(image),'video':str(video),'url':'http://127.0.0.1:18765/'+str(video.relative_to(Path('/root/gpufree-data'))),'visual_reviewed':False,'scope':'One preview clip per model only, not complete46 or visual acceptance'};rows.append(row);print(row['state'],row['url'],'frames',expected,'JPEG_bytes',image.stat().st_size)
(b/'early_video_audit.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'complete_delivery':False},indent=2))
