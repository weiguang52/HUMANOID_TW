"""Deliver final policies, matched diagnostics, and source/robot video index."""
import hashlib,html,json,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

state=Path('/root/gpufree-data/datasets/practice9/tw100_expression_v1')
val=state/'validation'
out=Path('validation_artifacts/tw100_validation')
if out.exists():raise FileExistsError(out)
for v in ('expression','control'):
    assert (val/v/'status').read_text().strip()=='completed'
d=json.loads((val/'comparison.json').read_text())
out.mkdir(parents=True)
shutil.copy2(val/'comparison.json',out/'comparison.json')
shutil.copy2(val/'manifest.json',out/'manifest.json')
shutil.copy2(val/'manifest.contacts.npz',out/'manifest.contacts.npz')
shutil.copytree(val/'reference_videos',out/'reference_videos')
enc=[]
for v in ('expression','control'):
    dst=out/v;dst.mkdir()
    ck=Path((state/v/'final_checkpoint').read_text().strip())
    shutil.copy2(ck,dst/ck.name)
    shutil.copytree(ck.parent/'params',dst/'params')
    for p in (val/v).glob('*.json'):shutil.copy2(p,dst/p.name)
    for p in (val/v).glob('*.camera.npz'):shutil.copy2(p,dst/p.name)
    shutil.copytree(val/v/'videos',dst/'videos')
for p in out.rglob('*.mp4'):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(p)],text=True))
    assert info['streams'][0]['codec_name']=='h264'
    assert int(info['streams'][0]['nb_frames'])>0
    enc.append(dict(file=str(p.relative_to(out)),**info))
assert len(enc)==9
(out/'encoding.json').write_text(json.dumps(enc,indent=2))
camera_checks=[]
for p in out.rglob('*.camera.npz'):
    a=np.load(p);camera=a['camera_origin'];reset=a['reset'].astype(bool)
    pair=~(reset[1:]|reset[:-1])
    vertical_steps=np.abs(np.diff(camera[:,2]))[pair]
    camera_checks.append(dict(file=str(p.relative_to(out)),frames=len(camera),
        reset_cuts=int(reset.sum()),
        max_vertical_step_m=float(vertical_steps.max()) if len(vertical_steps) else None))
(out/'camera_checks.json').write_text(json.dumps(camera_checks,indent=2))

variants=['control','expression'];clips=['000016','000124','000346']
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,key,label,mult in zip(axes,['wrist_error_m','contact_mismatch_fraction','foot_link_contact_speed_m_s'],['Wrist error (cm)','Contact mismatch (%)','Contact foot-link speed (cm/s)'],[100,100,100]):
    for offset,v in [(-.18,'control'),(.18,'expression')]:
        vals=[np.mean([r[key] for r in d['variants'][v]['replays'] if r['replay'].startswith(c)])*mult for c in clips]
        ax.bar(np.arange(3)+offset,vals,.36,label=v)
    ax.set_xticks(range(3),clips);ax.set_title(label)
axes[0].legend();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(16,10),layout='constrained')
for ax,key,title in zip(axes,['actual_joint_2_8hz_rms_deg','command_joint_2_8hz_rms_deg'],['Actual','Command']):
    for offset,v in [(-.18,'control'),(.18,'expression')]:
        rows=d['variants'][v]['replays'];names=rows[0]['joint_names']
        values=np.sqrt(np.mean(np.square([r[key] for r in rows]),axis=0))
        ax.barh(np.arange(len(names))+offset,values,.36,label=v)
    ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.legend()
    ax.set_title(title);ax.set_xlabel('2-8Hz RMS (degrees; includes intentional movement)')
fig.savefig(out/'joint_band.png',dpi=140);plt.close(fig)
text=['# TW100 stage-one standing validation','',d['note'],'',
'Common termination: anchor height/orientation + wrist height. The control policy is evaluated without its training-time foot-reference termination, to use the same criterion as expression.',
'No observation noise; physical startup randomization varies with evaluation seed42/123/2026. One training seed only.',
'Original NPY videos are at20fps/source speed. Robot videos are50fps: 000016/000124 approximately20s, 000346 approximately35.74s; retargeted references have longer duration. Videos are NOT phase synchronized. Robot cameras follow horizontal translation smoothly with fixed height; resets may cause cuts.',
'Five-second windows use simulator time, preserve short tails, and never join reset episodes. Raw force threshold is1N; no time warping changes reference contacts. Unknown coverage is reported.','']
for v in variants:
    x=d['variants'][v]
    text += [f'## {v}',f'Clean replays: {x["clean_replays"]}/9',json.dumps(x['mean'],indent=2),'']
text += ['## Verdict',
'New policy NOT accepted as improvement: clean7/9 vs9/9; wrist error4.23cm vs3.52cm.',
'Contact mismatch improves5.88% vs10.94%, but contact foot-link speed worsens1.64cm/s vs0.97cm/s.',
'Actual joint2-8Hz RMS0.224deg vs0.130deg; includes intentional motion.',
'Expression000346 fails wrist-height tracking in seed42 and2026; no anchor fall termination.',
'Anchor-height error also worsens2.43cm vs0.74cm, so failure is not attributed solely to arms.',
'Keep baseline. Only three validation clips and one training seed.',
'Videos use separate simulator starts and can fail at different times from batch metrics.',
'Expression000346 full video terminates and resets at28.42s; do not confuse reset with successful continuation.',
'Means cover executed samples; failed policies do not cover the same full reference as successful policies. Completion is reported separately.']
(out/'README.md').write_text('\n'.join(text))
parts=['<!doctype html><meta charset="utf-8"><title>TW100 standing validation</title><style>body{font:16px sans-serif;margin:32px;max-width:1500px;background:#f4f6f8;color:#152334}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}img{max-width:100%}pre{white-space:pre-wrap}</style>',
'<h1>TW100 / 站立与上肢表达验证</h1><p>三条验证动作 × 三个评估种子。原始人体视频与机器人片段的时间尺度不同，未做相位同步；不能直接按播放器同一秒比较姿态。</p>',
'<p>机器人视频：000016/000124为约20秒片段，000346为约35.74秒。定量回放覆盖完整参考长度。5秒窗口不跨重置。此报告是可解释诊断，不是HumanScore。</p>',
'<img src="comparison.png"><details><summary>逐关节2–8Hz运动幅度（包含有意动作）</summary><img src="joint_band.png"></details>']
parts.append('<section><h2>Verdict: retain baseline</h2>'
    '<p>New policy:7/9 clean vs9/9. Wrist error4.23cm vs3.52cm. '
    'Contact mismatch5.88% vs10.94%, but contact foot-link speed1.64cm/s vs0.97cm/s. '
    'Expression000346 fails wrist-height tracking in two seeds. '
    'Contact improvement alone is insufficient for acceptance. '
    'Video replays use separate simulator starts; failure times may differ. '
    'Expression000346 full video resets at28.42s after tracking failure.</p></section>')
for c in clips:
    parts.append(f'<section><h2>{c}</h2><div class="grid">')
    for title,path in [('原始 HumanML3D / 原速',f'reference_videos/{c}_source.mp4'),('原方案',f'control/videos/{c}.mp4'),('新方案',f'expression/videos/{c}.mp4')]:
        parts.append(f'<div><h3>{title}</h3><video controls preload="metadata" src="{path}"></video></div>')
    parts.append('</div></section>')
parts.append('<section><h2>协议与指标</h2><pre>'+html.escape('\n'.join(text))+'</pre></section>')
(out/'index.html').write_text('\n'.join(parts))
(out/'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='sha256.json'},indent=2))
print(out)
