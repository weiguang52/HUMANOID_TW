import ast
from pathlib import Path
old=Path('scripts/locomotion/eval_issue15.py').read_text()
tree=ast.parse(old)
repls=ast.literal_eval(next(x.value for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='replacements' for t in x.targets)))
new=[]
for before,after in repls:
 after=after.replace('issue15','issue16')
 if before=='args_cli, hydra_args = parser.parse_known_args()':
  after='parser.add_argument("--issue16_mode",default="mirror",choices=["original","identity","mirror","default"])\n'+after
 if before=='    for command_index, (label, command) in enumerate(commands):':
  after='    order = ["forward_03","yaw_positive","yaw_negative","forward_05","forward_08","stand"]\n    commands.sort(key=lambda x:order.index(x[0]))\n    for label, command in commands:\n        command_index = order.index(label)'
 if before=='    results = []':
  after=after.replace('args_cli.issue16_capture_command)','args_cli.issue16_capture_command,mode=args_cli.issue16_mode)')
 if before=='"checkpoint":resume_path,"deterministic_policy":True,':
  after=after.replace('"issue16_tracking":','"issue16_intervention":dict(mirror_metadata,calls=policy.calls),\n        "issue16_tracking":')
 new.append((before,after))
new.append(('    robot = base.scene["robot"]','    robot = base.scene["robot"]\n    from issue16_mirror import install_policy\n    policy, mirror_metadata = install_policy(policy,base,args_cli.issue16_mode,args_cli.issue16_records)'))
prefix=old[:old.index('replacements=')].replace('60a8cc25ccce3a8ae22fcf7e5dd6f8d53db7613b','770014ee9a4393da6dff34eced89a04d83e788dd').replace('Issue15 pure observational','Issue16 inference-only reflection and observational')
suffix=old[old.index('for old,new in replacements:'):].replace('issue15','issue16')
s=prefix+'replacements='+repr(new)+'\n'+suffix
p=Path('scripts/locomotion/eval_issue16.py');compile(s,str(p),'exec');p.write_text(s)
print('EVAL16_ADAPTER_COMPILES',len(new))
