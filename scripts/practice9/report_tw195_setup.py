import pathlib,json,shutil,hashlib,subprocess,html
S=pathlib.Path('/root/gpufree-data/datasets/practice9/tw195_v1');O=pathlib.Path('validation_artifacts/tw195_setup')
for f in ['data_audit.json','selection.json','final_audit.json','pilot_smooth_audit.json','tests.log']:
 shutil.copy2(S/f,O/f)
for v in ['original','corrected']:shutil.copy2(S/f'train_{v}.json',O/f'train_{v}.json')
e=json.load(open(O/'experiment.json'));e['smoothing']='Zero-phase Gaussian sigma2 frames on the 10 solved leg joints; root correction only smoothed, yaw/upper joints fixed; no contact time resampling. Full solved-angle smoothing prevents velocity overshoot from delta-only smoothing.';(O/'experiment.json').write_text(json.dumps(e,indent=2))
enc=[]
for p in O.rglob('*.mp4'):
 s=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,nb_read_frames,r_frame_rate','-of','json',str(p)],text=True))['streams'][0];assert s['codec_name']=='h264' and int(s['nb_read_frames'])>0;enc.append(dict(file=str(p.relative_to(O)),**s))
(O/'encoding.json').write_text(json.dumps(enc,indent=2))
text='''TW195 reference correction pilot. These URDF videos are kinematic replays, not trained controllers or physics validation. Each before/after pair is aligned on unchanged robot reference frames (displayed at25fps, every second50Hz frame). Source NPY videos use original20fps timing and are NOT synchronized with time-stretched robot reference. Human contact bits and confidence remain unchanged. Both ankle/toe source bits + known stance enable flat-foot constraints; partial contacts preserve foot tilt. Unknown states do not create support labels. Constant ground lift after smoothing <=3mm. No yaw or upper joint changes; pelvis height adjustments translate upper body. Foot geometry improvement alone does not prove dynamic stability. Some original clips violate upper joint velocity limits and are excluded equally from both training variants. Original treadmill/stairs evaluation references remain unchanged and out of flat-floor scope. Standing clip000346 retains original reference in both groups: its original right shoulder yaw velocity exceeds the URDF limit by11.25%, so it is a separately reported diagnostic, not qualified reference acceptance.'''
page='<!doctype html><meta charset="utf-8"><title>TW195 contact IK</title><style>body{max-width:1300px;margin:auto;font:18px sans-serif}video{width:100%}pre{white-space:pre-wrap}</style><h1>TW195 Contact-constrained reference IK</h1><p>'+text+'</p><pre>'+html.escape((S/'data_audit.json').read_text())+'</pre>'
for mid in ['000016','000832','000211']:
 page+=f'<h2>{mid}: original / corrected URDF</h2><video controls src="{mid}_before_after.mp4"></video><p>Original HumanML3D source timing:</p><video controls src="humanml_source/{mid}_source.mp4"></video>'
(O/'index.html').write_text(page)
files={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name!='sha256.json'};(O/'sha256.json').write_text(json.dumps(files,indent=2));print('verified videos',len(enc))
