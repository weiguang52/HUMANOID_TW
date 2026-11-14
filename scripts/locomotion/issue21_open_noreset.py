"""Prescribed200-step open plant response must continue physically after fall flags."""
import json,torch
from pathlib import Path
import issue21_runtime as runtime
from issue21_common import sha

def run(env,policy,runner,args,checkpoint):
 assert args.issue21_mode=='ground' and args.issue21_kind=='open';base=env.unwrapped;initialize=runtime.reset;requests=[]
 def initial_reset(*a,**kw):
  state=initialize(*a,**kw)
  original_compute=base.termination_manager.compute
  def compute_without_auto_reset(*a,**kw):
   flags=original_compute(*a,**kw);ids=flags.nonzero(as_tuple=False).flatten()
   if len(ids):requests.append({'at_common_step':int(base.common_step_counter),'environments':ids.detach().cpu().tolist()})
   return torch.zeros_like(flags)
  base.termination_manager.compute=compute_without_auto_reset
  return state
 runtime.reset=initial_reset
 runtime.run(env,policy,runner,args,checkpoint)
 assert int(base.episode_length_buf.min())==200 and int(base.episode_length_buf.max())==200
 p=Path(args.output);d=json.loads(p.read_text());d['diagnostic_lifecycle']={'automatic_resets_disabled_after_initialization':True,'termination_conditions_unchanged':True,'reward_and_physical_parameters_unchanged':True,'all_environment_episode_steps_at_end':200,'reset_requests_suppressed':requests,'one_continuous_physical_trajectory_per_env':True,'wrapper_source_sha256':sha(__file__)};p.write_text(json.dumps(d,indent=2)+'\n');print('CONTIGUOUS_OPEN_REPLAY_PASS200',len(requests),flush=True)
