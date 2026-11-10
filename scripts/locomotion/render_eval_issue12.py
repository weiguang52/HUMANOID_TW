"""Render-only derivative of frozen eval code: static warmup and honest labels."""
import hashlib,json,sys
from pathlib import Path
from issue12_common import ROOT

assert "--video" in sys.argv,"This adapter is only for illustrative videos"
path=Path("scripts/locomotion/eval_cmd.py").resolve()
source=path.read_text()
frozen=json.loads((ROOT/"frozen_training_sources.json").read_text())
original_sha=hashlib.sha256(path.read_bytes()).hexdigest()
assert original_sha==frozen["sha256"]["scripts/locomotion/eval_cmd.py"]
assert source.count("for _ in range(12):")==1
source=source.replace("for _ in range(12):","for _ in range(60):")
needle="                writer.append_data(frame)"
assert source.count(needle)==1
overlay='''                from PIL import Image, ImageDraw
                import numpy as np
                image = Image.fromarray(frame[:,:,:3])
                draw = ImageDraw.Draw(image)
                draw.rectangle((0,0,image.width,40),fill=(20,20,20))
                draw.text((10,5),"ILLUSTRATIVE ONE-ENV ROLLOUT | ACCEPTANCE FROM 92160 TRIALS",fill="white")
                draw.text((10,22),f"{Path(args_cli.output).stem} | {label} | {args_cli.camera_view} | t={(step+1)*base.step_dt:.2f}s",fill="white")
                writer.append_data(np.asarray(image))'''
source=source.replace(needle,overlay)
identity={"original_eval_sha256":original_sha,
          "in_memory_render_eval_sha256":hashlib.sha256(source.encode()).hexdigest(),
          "static_first_frame_warmup_renders":60,"physics_step_calls_or_configuration_changes":False,
          "other_changes":"Only two text substitutions: static render warmup12->60 and video labels.",
          "video_scope":"Independent illustrative one-env single-command seed1; paired reset index0. Not an exact1024-env failure or acceptance sample.",
          "camera":"follows translation and heading"}
p=ROOT/"render_source_identity.json"
if p.exists():assert json.loads(p.read_text())==identity
else:p.write_text(json.dumps(identity,indent=2))
sys.argv[0]=str(path)
__file__=str(path)
exec(compile(source,str(path),"exec"),globals(),globals())
