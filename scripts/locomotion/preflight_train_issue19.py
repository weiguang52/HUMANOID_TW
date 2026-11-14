"""Two-update runner gate; compare new disabled arm against exact prior train.py."""
import os, subprocess
from pathlib import Path
mode=os.environ['TW_ISSUE19_PREFLIGHT_MODE']
assert mode in ['baseline','sym_cont','sym_wrap']
root=Path('/root/gpufree-data/tw_stage1/issue19_20261113/preflight')/mode
root.mkdir(parents=True,exist_ok=True)
s=subprocess.check_output(['git','show','91428e3c316eae754fc8aa557c254ca21eb71742:scripts/locomotion/train.py'],text=True) if mode=='baseline' else Path('scripts/locomotion/train.py').read_text()
needle='        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)'
add=needle+"""
        runner.save(str(__import__('pathlib').Path(os.environ['TW_ISSUE19_PREFLIGHT_OUTPUT'])/'final.pt'))
        import random,numpy as np
        torch.save({'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'root':env.unwrapped.scene['robot'].data.root_state_w.clone(),'joint_pos':env.unwrapped.scene['robot'].data.joint_pos.clone(),'joint_vel':env.unwrapped.scene['robot'].data.joint_vel.clone()},str(__import__('pathlib').Path(os.environ['TW_ISSUE19_PREFLIGHT_OUTPUT'])/'runtime.pt'))
"""
assert s.count(needle)==1
s=s.replace(needle,add)
(root/'instrumented_train.py').write_text(s)
os.environ['TW_ISSUE19_PREFLIGHT_OUTPUT']=str(root)
exec(compile(s,'scripts/locomotion/train.py','exec'),{'__name__':'__main__','__file__':str(Path('scripts/locomotion/train.py').resolve())})
