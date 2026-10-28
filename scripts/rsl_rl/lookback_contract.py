from pathlib import Path
import json

def prepare(cfg,log_dir,checkpoint=None):
 c=cfg.commands.motion;share=float(c.adaptive_lookback_share)
 if not 0<=share<=1:raise ValueError('Invalid lookback share')
 payload={'version':1,'share':share,'prior_ratio':float(c.adaptive_uniform_ratio),'bin_size_s':float(c.adaptive_bin_size_s),'kernel_size':int(c.adaptive_kernel_size),'semantics':'observed failure score split to immediately preceding same-motion bin; unvisited unchanged','sampler_statistics_resume':'not persisted; counters reset on environment construction'}
 if share and c.adaptive_kernel_size!=1:raise ValueError('Lookback requires kernel1')
 if checkpoint:
  p=Path(checkpoint).parent/'params/lookback_sampling.json'
  if p.exists() and json.loads(p.read_text())!=payload:raise ValueError('Resume sampling contract mismatch')
 if share:
  p=Path(log_dir)/'params/lookback_sampling.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(payload,indent=2))
 return payload if share else None

def finish(env,log_dir,payload):
 if payload is None:return
 c=env.unwrapped.command_manager.get_term('motion');calls=getattr(c,'_lookback_calls',0)
 if not calls:raise ValueError('No lookback calls observed')
 out={'contract':payload,'calls':calls,'changed_calls':getattr(c,'_lookback_changed_calls',0)}
 (Path(log_dir)/'lookback_sampling_completed.json').write_text(json.dumps(out,indent=2))
