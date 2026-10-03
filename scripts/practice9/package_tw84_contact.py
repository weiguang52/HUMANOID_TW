"""Archive completed contact ablation; never replace older delivery."""
import json,shutil,hashlib,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/root/gpufree-data/datasets/practice9')
state=root/'tw75_contact_v1';out=Path('validation_artifacts/tw84_contact_v1')
out.mkdir(parents=True,exist_ok=True)
d=json.loads((root/'tw91_slip_v1/diagnosis/diagnosis.json').read_text())
shutil.copy2(root/'tw91_slip_v1/diagnosis/diagnosis.json',out/'diagnosis.json')
shutil.copy2(state/'comparison.json',out/'comparison.json')
encoding=[]
for v in ['control','contact']:
 src=state/v;dst=out/v;dst.mkdir(exist_ok=True)
 ck=Path((src/'final_checkpoint').read_text().strip())
 shutil.copy2(ck,dst/ck.name)
 shutil.copytree(ck.parent/'params',dst/'params',dirs_exist_ok=True)
 for p in src.glob('*.json'):shutil.copy2(p,dst/p.name)
 (dst/'videos').mkdir(exist_ok=True)
 for p in sorted((src/'videos').glob('*.mp4')):
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate','-of','json',str(p)],text=True))
  encoding.append(dict(variant=v,clip=p.stem,streams=info['streams']))
  if p.stem in ['000039','000139','000211']:shutil.copy2(p,dst/'videos'/p.name)
 for p in src.glob('*.camera.npz'):shutil.copy2(p,dst/p.name)
(out/'encoding.json').write_text(json.dumps(encoding,indent=2))
fig,axes=plt.subplots(1,2,figsize=(15,12),layout='constrained')
for offset,v in [(-.25,'baseline'),(0,'control'),(.25,'contact')]:
 rr=[r for r in d['replays'] if r['variant']==v and r['clip'][:6] in ['000039','000139','000211']]
 names=rr[0]['joint_names'];y=np.arange(len(names))
 for ax,key,title in zip(axes,['target_q','q'],['Command','Actual']):
  b=np.sqrt(np.mean([np.square(r['band_rms'][key]) for r in rr],axis=0))
  ax.barh(y+offset,b,height=.25,label=v)
  ax.set_yticks(y,names);ax.set_xlabel('2-8 Hz RMS (degrees)');ax.set_title(title)
  ax.invert_yaxis() if offset==-.25 else None
axes[0].legend();fig.suptitle('Three validation walking clips x three seeds; reset boundaries excluded')
fig.savefig(out/'joint_band.png',dpi=120);plt.close(fig)
report='''TW84 contact ablation completed, improvement NOT accepted.
Both variants: 4000 extra PPO iterations from same full-corpus seed123 checkpoint.
Contact phase cost: control 0 vs contact -0.5; other rewards/controller unchanged.
All 36 evaluation reports and 12 videos generated. Six walking videos archived here,
including failed 000139. Standing videos and raw telemetry remain on server.
Clean replay completion: original baseline15/18, control13/18, contact15/18.
Contact vs control: walking slip -9%, raw force-threshold switching +20%; standing
switching -26%. These are validation results, not independent test acceptance.
Raw switching is not equivalent to full foot flight. Band RMS includes intentional
motion; it is not a universal jitter score. Baseline was not retrained in this round.
Original HumanML contact bits retained. Confidence masking yields no trusted stance
for treadmill000039; some robot FK reference stance feet remain 1-2cm above ground.
Reference labels were not changed to improve these validation scores.
Next experiment TW91 varies EXISTING feet_slide weight -0.2 vs -1.0, not a duplicate.
Video frame sampling of000139 shows low foot clearance/forward lean; isolated frames
cannot establish high-frequency motion quality. Full perceptual review remains limited.
Camera follows horizontal translation only, tau0.5, fixed direction/height; reset cuts.
Keep final weights alongside params/control_runtime.json to preserve runtime behavior.
'''
(out/'README.txt').write_text(report)
(out/'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='sha256.json'},indent=2))
print(out)
