"""Extend frozen issue19 evaluator with evaluation-only phase/reset/USD factors."""
from pathlib import Path
previous=Path(__file__).with_name('eval_issue19.py');source=previous.read_text()
source=source.replace('root=Path("/root/gpufree-data/tw_stage1/issue19_20261113")','root=Path("/root/tw_stage1/issue20_20261114")')
extra=[
 ('args_cli, hydra_args = parser.parse_known_args()', 'parser.add_argument("--issue20_phase",type=int,choices=[0,10],default=0)\nparser.add_argument("--issue20_usd",default="")\nparser.add_argument("--issue20_baseline",action="store_true")\nparser.add_argument("--issue20_default",action="store_true")\nargs_cli, hydra_args = parser.parse_known_args()'),
 ('    env = gym.make(', '    from issue20_runtime import configure\n    if not args_cli.issue20_baseline: configure(env_cfg,args_cli.issue20_phase,args_cli.issue20_usd)\n    else: env_cfg.issue20_phase_offset=0\n    env = gym.make('),
 ('    cmd_term = base.command_manager.get_term("base_velocity")', '    from issue20_runtime import install_resets\n    initial_metadata=install_resets(base,args_cli.issue20_phase,args_cli.issue18_records)\n    cmd_term = base.command_manager.get_term("base_velocity")'),
 ('    results = []', '    if args_cli.issue20_default:\n        assert args_cli.command_filter == "stand"\n        policy.policy=lambda obs: torch.zeros((base.num_envs,30),device=base.device)\n    results = []'),
 ('"physics_diagnostics":physics_diagnostics,', '"physics_diagnostics":physics_diagnostics,"issue20":dict(initial_metadata,usd=args_cli.issue20_usd,baseline=args_cli.issue20_baseline,default_PD=args_cli.issue20_default),')]
marker='for old,new in replacements:';assert source.count(marker)==1
source=source.replace(marker,'replacements.extend(_EXTRA)\n'+marker).replace('from issue19_tracking import YawRecorder','from issue20_tracking import YawRecorder')
assert '/root/gpufree-data/tw_stage1/issue19_20261113' not in source
exec(compile(source,str(previous),'exec'),{'__name__':'__main__','__file__':str(previous),'_EXTRA':extra})
