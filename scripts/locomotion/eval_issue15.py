"""Issue15 pure observational adapter around the unchanged issue13/14 evaluation source."""
import difflib,hashlib,subprocess,sys
from pathlib import Path
BASE="60a8cc25ccce3a8ae22fcf7e5dd6f8d53db7613b"
original=Path(__file__).with_name("eval_cmd.py")
source=original.read_text()
assert original.read_bytes()==subprocess.check_output(["git","show",BASE+":scripts/locomotion/eval_cmd.py"])
source_sha=hashlib.sha256(original.read_bytes()).hexdigest()
replacements=[
('args_cli, hydra_args = parser.parse_known_args()', 'parser.add_argument("--critic_recovery_clock",action="store_true",default=False)\nparser.add_argument("--issue15_records",default="")\nparser.add_argument("--issue15_capture_env",type=int,default=-1)\nparser.add_argument("--issue15_capture_command",default="forward_03")\nargs_cli, hydra_args = parser.parse_known_args()'),
('    if args_cli.video:\n        if env_cfg.scene.num_envs', '    if args_cli.critic_recovery_clock:\n        from unitree_rl_lab.tasks.locomotion.robots.tw.critic_recovery_clock import configure_critic_recovery_clock\n        configure_critic_recovery_clock(env_cfg,True,evaluation=True)\n    if args_cli.video:\n        if env_cfg.scene.num_envs'),
('    results = []', '    tracking = None\n    if args_cli.issue15_records:\n        from issue15_tracking import YawRecorder\n        tracking = YawRecorder(base,runner,args_cli.issue15_records,resume_path,args_cli.issue5_condition,args_cli.issue15_capture_env,args_cli.issue15_capture_command)\n    results = []'),
('    for command_index, (label, command) in enumerate(commands):','    if args_cli.issue15_records:\n        order = ["forward_03","yaw_positive","yaw_negative","forward_05","forward_08","stand"]\n        commands.sort(key=lambda x:order.index(x[0]))\n    for command_index, (label, command) in enumerate(commands):'),
('        writer = None', '        if tracking is not None:\n            tracking.begin(label,command)\n        writer = None'),
('            count += valid.sum()', '            if tracking is not None:\n                tracking.sample(values,valid,robot.data.root_lin_vel_b,robot.data.root_ang_vel_b[:,2])\n            count += valid.sum()'),
('        print("EVAL",json.dumps(results[-1]),flush=True)', '        if tracking is not None:\n            tracking.finish(results[-1])\n        print("EVAL",json.dumps(results[-1]),flush=True)'),
('"checkpoint":resume_path,"deterministic_policy":True,','"checkpoint":resume_path,"deterministic_policy":True,\n        "issue15_tracking":None if tracking is None else tracking.job_metadata(),\n        "issue15_source":{"original_sha256":_ORIGINAL_SHA,"instrumented_sha256":_PATCHED_SHA},')
]
for old,new in replacements:
 assert source.count(old)==1,old
 source=source.replace(old,new)
patched_sha=hashlib.sha256(source.encode()).hexdigest()
root=Path("/root/gpufree-data/tw_stage1/issue15_20261111")
(root/"adapter_source_diff.patch").write_text("".join(difflib.unified_diff(original.read_text().splitlines(True),source.splitlines(True),fromfile="original_eval_cmd.py",tofile="read_only_issue15_adapter.py")))
sys.argv[0]=str(original)
exec(compile(source,str(original),"exec"),{"__name__":"__main__","__file__":str(original),"_ORIGINAL_SHA":source_sha,"_PATCHED_SHA":patched_sha})
