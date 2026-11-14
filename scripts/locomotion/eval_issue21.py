"""Reuse exact baseline environment/policy setup; replace only diagnostic loop."""
from pathlib import Path
import subprocess
p=Path(__file__).with_name('eval_cmd.py');s=p.read_text();assert p.read_bytes()==subprocess.check_output(['git','show','b8ddc926:scripts/locomotion/eval_cmd.py'])
s=s.replace('args_cli, hydra_args = parser.parse_known_args()', 'parser.add_argument("--issue21_mode",choices=["ground","air","perturb"],required=True)\nparser.add_argument("--training_seed",type=int,required=True)\nparser.add_argument("--issue21_kind",choices=["closed","open"],default="closed")\nparser.add_argument("--issue21_phase",type=int,choices=[0,10],default=0)\nparser.add_argument("--issue21_command",choices=["stand","forward_05"],required=True)\nparser.add_argument("--issue21_epsilon",type=float,default=0.0)\nargs_cli, hydra_args = parser.parse_known_args()')
s=s.replace('    env = gym.make(', '    from issue21_runtime import configure\n    configure(env_cfg,args_cli.issue21_mode)\n    env = gym.make(')
s=s.replace('    runner.load(resume_path)', '    from issue19_actor import install_actor\n    from issue17_symmetry import mapping_for_env\n    install_actor(runner.alg.actor,mapping_for_env(env.unwrapped),runner.alg.optimizer)\n    runner.load(resume_path)')
a=s.index('    # One independent, full-duration trial per environment and command.');b=s.index('\n\nif __name__ == "__main__":',a)
s=s[:a]+'    from issue21_runtime import run\n    run(env,policy,runner,args_cli,resume_path)\n    env.close()\n'+s[b:]
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
