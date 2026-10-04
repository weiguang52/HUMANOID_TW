"""Per-category 5s validation diagnostics and source/robot video delivery."""
import sys,json,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
from summarize_tw100_validation import summarize

variant=sys.argv[1]
assert variant in ('step','velocity')
state=Path('/root/gpufree-data/datasets/practice9/tw140_v1')
root=state/'validation';src=root/variant
manifest=json.loads((root/'manifest.json').read_text())
categories={r['id']:r['category'] for r in manifest['motions']}
paths=sorted(p for p in src.glob('*.seed*.json') if not p.name.endswith('.summary.json'))
assert len(paths)==18,(variant,len(paths))
rows=[]
for p in paths:
    row=summarize(p);row['category']=categories[p.name.split('.')[0]]
    with np.load(p.with_suffix('.npz')) as a:
        row['joint_target_error_rms_rad']=np.sqrt(np.mean((a['target_q']-a['q'])**2,axis=0)).tolist()
    from step_diagnostics import diagnose
    motion=next(r for r in manifest['motions'] if r['id']==p.name.split('.')[0])
    offset=sum(r['frames'] for r in manifest['motions'][:manifest['motions'].index(motion)])
    row['step_diagnostics']=diagnose(p,motion,offset)
    rows.append(row)
result=dict(variant=variant,training_seed=42,evaluation_seeds=[42,123,2026],rows=rows,
    protocol='Six held-out validation clips; not full test-set acceptance. 5s continuous windows retain short tails and reset boundaries; not HumanScore. Unknown contact excluded with coverage. Foot-link speed includes rolling. Joint 2-8Hz energy includes intentional movement. Torque is implicit-PD estimate, not measured motor torque.')
result['categories']={}
for cat in sorted(set(categories.values())):
    rr=[r for r in rows if r['category']==cat]
    result['categories'][cat]=dict(clean=sum(r['clean_motion_end'] for r in rr),total=len(rr),
        metrics={k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in
        ['upper_error_m','wrist_error_m','anchor_height_error_m','contact_mismatch_fraction',
         'foot_link_contact_speed_m_s','contact_known_fraction','worst_window_wrist_error_m']})
(src/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
# Separate output per variant avoids concurrent report writes. Final files only.
dst=Path('validation_artifacts/tw140_mixed')/variant
dst.mkdir(parents=True,exist_ok=True)
checkpoint=Path((state/variant/'final_checkpoint').read_text().strip())
shutil.copy2(checkpoint,dst/'model_final.pt')
shutil.copytree(checkpoint.parent/'params',dst/'params',dirs_exist_ok=True)
for name in ('source_commit','data_audit.json'):
    shutil.copy2(state/variant/name,dst/name)
shutil.copy2(src/'comparison.json',dst/'comparison.json')
shutil.copy2(root/'manifest.json',dst/'evaluation_manifest.json')
for p in paths:shutil.copy2(p,dst/p.name)
shutil.copytree(src/'videos',dst/'videos',dirs_exist_ok=True)
camera_audit=[]
for p in src.glob('*.camera.npz'):
    shutil.copy2(p,dst/p.name)
    with np.load(p) as a:
        continuous=~(a['reset'][1:] | a['reset'][:-1])
        delta=np.diff(a['camera_origin'],axis=0)[continuous]
        vertical=float(np.max(np.abs(delta[:,2]))) if len(delta) else 0.
        assert vertical<1.e-6,(p,vertical)
        camera_audit.append(dict(file=p.name,frames=len(a['reset']),
            continuous_max_vertical_step_m=vertical,
            continuous_max_horizontal_step_m=float(np.linalg.norm(delta[:,:2],axis=1).max()) if len(delta) else 0.))
assert len(camera_audit)==6,len(camera_audit)
(dst/'camera_audit.json').write_text(json.dumps(camera_audit,indent=2)+'\n')
ref=dst/'humanml_source'
subprocess.run([sys.executable,'scripts/practice9/render_validation_references.py',
    '--manifest',str(root/'manifest.json'),'--output',str(ref)],check=True)
encoding=[]
for p in sorted(dst.rglob('*.mp4')):
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
        '-select_streams','v:0','-show_entries','stream=codec_name,nb_read_frames,r_frame_rate',
        '-of','json',str(p)],text=True))['streams'][0]
    assert meta['codec_name']=='h264' and int(meta['nb_read_frames'])>0,(p,meta)
    if p.parent.name=='videos':
        motion=next(r for r in manifest['motions'] if r['id']==p.stem)
        requested=max(1000,motion['frames']+1)
        # Recorder may include the initial frame; allow exactly one control frame.
        assert int(meta['nb_read_frames']) in (requested-1,requested),(p,meta,requested)
        numerator,denominator=map(float,meta['r_frame_rate'].split('/'))
        assert abs(numerator/denominator-50)<1.e-6
    encoding.append(dict(file=str(p.relative_to(dst)),**meta))
assert len(encoding)==12,len(encoding)
(dst/'encoding.json').write_text(json.dumps(encoding,indent=2)+'\n')
items=[]
for cid in categories:
    candidates=list(ref.rglob(f'*{cid}*.mp4'))
    assert len(candidates)==1,(cid,candidates)
    source=candidates[0].relative_to(dst).as_posix()
    items.append(f'<h2>{cid} / {categories[cid]}</h2><div class="pair">'
        f'<video controls src="{source}"></video><video controls src="videos/{cid}.mp4"></video></div>')
html='''<!doctype html><meta charset="utf-8"><title>TW140 mixed validation</title>
<style>body{max-width:1400px;margin:auto;font:18px sans-serif}.pair{display:flex}video{width:49%}pre{white-space:pre-wrap}</style>
<h1>TW140 mixed walking / standing validation</h1>
<p>Left: original HumanML3D NPY (20fps). Right: robot simulation (50fps), stable side camera.
Different reference durations: these videos are not phase aligned. No source-contact time warping.</p>
<p>Metrics: 3 evaluation seeds, one training seed; mean and worst 5s windows in comparison.json.
Incomplete/failing rollouts must not be treated as complete reference coverage.</p>'''
(dst/'index.html').write_text(html+f'<pre>{json.dumps(result["categories"],indent=2)}</pre>'+''.join(items))
sha={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest()
     for p in dst.rglob('*') if p.is_file() and p.name!='sha256.json'}
(dst/'sha256.json').write_text(json.dumps(sha,indent=2)+'\n')
print(json.dumps(result['categories'],indent=2))
