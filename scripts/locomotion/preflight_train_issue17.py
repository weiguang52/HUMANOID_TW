"""Run exactly two validation updates with baseline or new disabled entry point."""
import os,subprocess
from pathlib import Path
mode=os.environ['TW_ISSUE17_PREFLIGHT_MODE'];assert mode in ['original','disabled','control_int','sym_int']
p=Path('/root/gpufree-data/tw_stage1/issue17_20261113/preflight')/mode;p.mkdir(parents=True,exist_ok=True)
s=subprocess.check_output(['git','show','c073475561787be7b81a87f4fa7af7fc46fd86de:scripts/locomotion/train.py']).decode() if mode=='original' else Path('scripts/locomotion/train.py').read_text()
old='        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)'
new=old+'''\n        runner.save(str(__import__('pathlib').Path(os.environ['TW_ISSUE17_PREFLIGHT_OUTPUT'])/'final.pt'))
        import random,numpy as np
        torch.save({'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'root':env.unwrapped.scene['robot'].data.root_state_w.clone(),'joint_pos':env.unwrapped.scene['robot'].data.joint_pos.clone(),'joint_vel':env.unwrapped.scene['robot'].data.joint_vel.clone()},str(__import__('pathlib').Path(os.environ['TW_ISSUE17_PREFLIGHT_OUTPUT'])/'runtime.pt'))
'''
if mode in ['control_int','sym_int']:
    check='''        from issue17_symmetry import mapping_for_env,data_augmentation_func
        import json
        mapping_for_env(env)
        observations=env.get_observations()
        assert observations['policy'].shape==(4096,101) and observations['critic'].shape==(4096,134)
        assert env.unwrapped.cfg.issue17_integer_clock
        augmented,_=data_augmentation_func(env,observations,None)
        assert augmented.batch_size[0]==8192
        __import__('pathlib').Path(os.environ['TW_ISSUE17_PREFLIGHT_OUTPUT'],'enabled_contract.json').write_text(json.dumps(dict(mode=os.environ['TW_ISSUE17_PREFLIGHT_MODE'],actor_dim=101,critic_dim=134,rollout_batch=4096*24,original_minibatch=4096*24//4,augmented_minibatch=4096*24//4*2,symmetry_active=runner.alg.symmetry is not None,integer_clock=True)))
'''
    new=check+new
assert s.count(old)==1;s=s.replace(old,new);(p/'instrumented_train.py').write_text(s);os.environ['TW_ISSUE17_PREFLIGHT_OUTPUT']=str(p)
exec(compile(s,'scripts/locomotion/train.py','exec'),{'__name__':'__main__','__file__':str(Path('scripts/locomotion/train.py').resolve())})
