"""Check observer neutrality against original evaluator with final physical state/full RNG."""
import json,os
import numpy as np
from issue15_common import ROOT,REPO,run,evaluate_args,sha
def main():
 os.chdir(REPO);cohort=json.loads((ROOT/"cohort.json").read_text());folder=ROOT/"smoke";folder.mkdir(exist_ok=True)
 checks={}
 for name in ["plain","observed","clock"]:
  m=cohort["models"]["clock_s1" if name=="clock" else "ft5_s1"]
  out=folder/(name+".json")
  args=evaluate_args(m,1,out,num_envs=64,commands=["stand","forward_03"] if name=="clock" else ["forward_03"],observer=name!="plain",plain=name=="plain")
  args+=["--observation_audit",str(folder/(name+"_state"))]
  if not out.exists():run(args,out.with_suffix(".log"),"smoke_"+name)
  data=json.loads(out.read_text());checks[name]={"passed":True,"checkpoint_sha256":sha(m["checkpoint"]),"commands":len(data["results"]),"formal_trials":0,"training_updates":0}
 a=json.loads((folder/"plain.json").read_text());b=json.loads((folder/"observed.json").read_text());assert a["results"]==b["results"]
 a=dict(np.load(folder/"plain_state/forward_03.npz"));b=dict(np.load(folder/"observed_state/forward_03.npz"));assert a.keys()==b.keys()
 for k in a:assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),k
 for name in ["observed","clock"]:
  d=json.loads((folder/(name+".json")).read_text());assert d["issue15_tracking"]["model_optimizer_state_unchanged"]
  assert d["observation_dims"]=={"policy":[101],"critic":[136 if name=="clock" else 134]}
 (ROOT/"smoke_audit.json").write_text(json.dumps({"passed":True,"checks":checks,"original_metrics_exact":True,"final_root_joint_state_and_all_rng_bitwise_identical":True,"compared_state_fields":list(a),"formal_updates":0,"formal_trials":0},indent=2)+"\n")
 print("ISSUE15_SMOKE_PASS",flush=True)
if __name__=="__main__":main()
