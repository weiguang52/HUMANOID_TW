"""Run the issue16 geometry/axis inspection on the derived strict model."""
from pathlib import Path
p=Path(__file__).with_name('inspect_issue16_usd.py');s=p.read_text();s=s.replace("R=Path('/root/gpufree-data/tw_stage1/issue16_20261111')","R=Path('/root/tw_stage1/issue20_20261114')");s=s.replace("P=Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3.usd')","P=Path('/root/tw_stage1/issue20_20261114/strict_symmetric.usd')");exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
