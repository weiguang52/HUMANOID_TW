"""Explicit experiment switch; legacy is a true configuration no-op."""
def configure_issue17(env_cfg,agent_cfg,mode):
    assert mode in ['legacy','control_int','sym_int']
    if mode=='legacy':return
    assert not getattr(agent_cfg.algorithm,'symmetry_cfg',None)
    env_cfg.issue17_integer_clock=True
    if mode=='sym_int':
        agent_cfg.algorithm.symmetry_cfg=dict(use_data_augmentation=True,use_mirror_loss=False,
            data_augmentation_func='issue17_symmetry:data_augmentation_func',mirror_loss_coeff=0.0)
