"""Read the complete fixed clean primary cohort while combined jobs continue."""
import json
from pathlib import Path
import numpy as np
from issue14_common import ROOT,REPO,sha
from run_issue14 import verify_frozen
from issue14_analysis_data import load_command,pooled_statistics
from issue13_statistics import compare

def main():
 verify_frozen();jobs=json.loads((ROOT/"formal_job_state.json").read_text())
 assert sum(v["condition"]=="clean" and v["passed"] for v in jobs.values())==15
 old=Path("/root/gpufree-data/tw_stage1/issue13_20261110/paired_comparisons.json")
 archive=REPO/"validation_artifacts/stage1_issue13_20261110"
 manifest=json.loads((archive/"sha256.json").read_text());assert sha(old)==manifest["paired_comparisons.json"]==sha(archive/"paired_comparisons.json")
 prior=json.loads(old.read_text())["clean:forward_08:full_horizon_nonterminal:all:xy_error"]
 new=[];inputs=[]
 models=json.loads((ROOT/"final_models.json").read_text())
 for seed in range(1,6):
  values=[]
  for es in [1,2,3]:
   name=f"clock_s{seed}";out=ROOT/"clean"/f"{name}_eval{es}.json";d=json.loads(out.read_text());m=models[name]
   assert sha(m["checkpoint"])==m["sha256"] and d["issue13_tracking"]["model_optimizer_state_unchanged"]
   row=next(x for x in d["results"] if x["command"]=="forward_08");folder=Path(d["issue13_tracking"]["output"])
   z,meta=load_command(folder,"forward_08",row);assert meta["num_envs"]==1024 and meta["steps"]==1000
   values.append(pooled_statistics(z,0,0,row["target"])["xy_error"])
   inputs.append({"model":name,"eval_seed":es,"raw_npz_sha256":sha(folder/"forward_08.npz"),"source_eval_sha256":sha(out),"independent_raw_mask_moment_and_original_metric_audit_passed":True})
  new.append(float(np.mean(values)))
 result=compare(prior["treatment"]["per_seed"],new)
 record={"passed":True,"primary_clean_dataset_complete":True,"overall_acceptance_still_pending_combined":True,
 "fixed_protocol_unchanged":True,"new_training_updates":0,"reference_group":"original_treatment","candidate_group":"critic_clock","training_seeds":5,"evaluation_jobs":15,
 "paired_bootstrap95_unadjusted":True,"primary":result,"primary_mitigation_supported":result["paired_difference"]["bootstrap95"][1]<0,
 "inputs":inputs,"prior_comparisons_sha256":sha(old),"source_sha256":{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name("issue14_analysis_data.py"),Path(__file__).with_name("issue13_statistics.py")]}}
 (ROOT/"primary_clean_complete_snapshot.json").write_text(json.dumps(record,indent=2))
 print(json.dumps({"primary_mitigation_supported":record["primary_mitigation_supported"],"old_treatment_mean":result["control"]["mean"],"clock_mean":result["treatment"]["mean"],"difference":result["paired_difference"]["mean"],"paired95":result["paired_difference"]["bootstrap95"],"five_training_seed_values":new},indent=2))
if __name__=="__main__":main()
