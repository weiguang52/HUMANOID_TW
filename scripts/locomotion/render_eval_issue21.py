"""Use exact selected precolored USD and saved states; never rerun dynamics."""
import json,sys,subprocess
from pathlib import Path
p=Path(__file__).with_name('eval_cmd.py');source=p.read_text();assert p.read_bytes()==subprocess.check_output(['git','show','c073475561787be7b81a87f4fa7af7fc46fd86de:scripts/locomotion/eval_cmd.py'])
selection=Path(sys.argv[sys.argv.index('--trajectory_replay')+1]);s=json.loads(selection.read_text());assets=json.loads((selection.parent/'colored_asset_integrity.json').read_text());assert assets['passed']
source=source.replace('from replay_failure import replay','from replay_issue21 import replay').replace('choices=["front","side"]','choices=["front","side","overhead"]')
a='/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3';source=source.replace(a+'_colored.usd',assets['colored_usd']).replace(a+'.usd',assets['physics_usd']);assert '--video' in sys.argv
sys.argv[0]=str(p);exec(compile(source,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
