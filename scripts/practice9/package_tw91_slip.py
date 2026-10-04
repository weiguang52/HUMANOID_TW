"""Archive final TW91 results remotely, including failures and final weights."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

state = Path('/root/gpufree-data/datasets/practice9/tw91_slip_v1')
out = Path('validation_artifacts/tw91_slip_v1')
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
encoding = []
for variant in ('control', 'slip'):
    for seed in (42, 123):
        src = state / variant / f'seed{seed}'
        assert (src / 'status').read_text().strip() == 'completed'
        dst = out / variant / f'seed{seed}'
        dst.mkdir(parents=True)
        ck = Path((src / 'final_checkpoint').read_text().strip())
        assert ck.name == 'model_43593.pt' and ck.stat().st_size > 0
        shutil.copy2(ck, dst / ck.name)
        shutil.copytree(ck.parent / 'params', dst / 'params')
        for p in src.glob('*.json'):
            shutil.copy2(p, dst / p.name)
        (dst / 'videos').mkdir()
        for p in sorted((src / 'videos').glob('*.mp4')):
            info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,r_frame_rate','-of','json',str(p)], text=True))
            s = info['streams'][0]
            assert s['codec_name'] == 'h264' and int(s['nb_frames']) == 999
            encoding.append(dict(variant=variant, seed=seed, clip=p.stem, stream=s))
            if p.stem in ('000039','000139','000211'):
                shutil.copy2(p, dst / 'videos' / p.name)
        for p in src.glob('*.camera.npz'):
            shutil.copy2(p, dst / p.name)
assert len(encoding) == 24
shutil.copy2(state / 'comparison.json', out / 'comparison.json')
(out / 'encoding.json').write_text(json.dumps(encoding, indent=2))
(out / 'README.md').write_text('''# TW91 final paired-seed slip ablation

Existing feet_slide weight: control -0.2, slip -1.0; contact phase -0.5 in both.
Each variant: training seeds 42/123, 4000 additional PPO iterations from the same TW84 checkpoint.
Seed123 resumed after shutdown; final model_43593.pt retained with runtime params.
72 evaluations (6 validation clips x 3 evaluation seeds x 4 policies), 24 videos verified.
12 walking videos included, including failed clips. Original NPY videos: ../tw98_reference_videos.
Robot videos: 50fps, 999 frames. Original NPY: 20fps; NOT phase-synchronized.

Standing: aggregate slip about -33%, body error -25%, 2-8Hz actual-joint RMS -43%.
Walking: slip about -3%, force-threshold switching -22%, joint RMS about +8%.
Clean walking replays: control 8/18, slip 7/18. Standing both 18/18.
000039 clean: control 3/6, slip 1/6; 000139 both 0/6; 000211 5/6 vs 6/6.
Conclusion: retain standing candidate; walking improvement NOT accepted. Baseline preserved.
Band RMS includes intentional movement; force-threshold switches do not prove foot flight.
These clips are validation, not independent test. Body-average error is not hand-expression fidelity.
Some recovery videos used performance rendering after GPU rendering failure; physics unchanged.
''')
(out / 'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name != 'sha256.json'}, indent=2))
print(out)
