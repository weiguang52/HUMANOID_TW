"""Freeze exact issue13 observer/statistical sources after runtime equivalence, before formal data."""
import json,subprocess,time
from pathlib import Path
from issue13_common import ROOT,REPO,BASE,sha
def main():
 assert (ROOT/"runtime_smoke_complete").exists()
 assert not any((ROOT/"clean").glob("*_eval*.json")) and not any((ROOT/"combined").glob("*_eval*.json"))
 assert subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()==BASE
 old=json.loads((REPO/"validation_artifacts/stage1_issue12_20261110/frozen_training_sources.json").read_text())
 files=list(old["sha256"])
 files += [str(p) for p in Path("scripts/locomotion").glob("*issue13*.py")]
 files += ["tests/locomotion/test_issue13_tracking.py","docs/stage1_issue13_20261110.md",
           str(ROOT/"protocol_plan.md"),str(ROOT/"input_models.json"),str(ROOT/"adapter_source_diff.patch")]
 for name in old["sha256"]:
  if not Path(name).is_absolute():
   assert Path(name).read_bytes()==subprocess.check_output(["git","show",BASE+":"+name]),name
 smoke=json.loads((ROOT/"runtime_smoke_audit.json").read_text());assert set(smoke)=={"clean","combined"} and all(x["passed"] for x in smoke.values())
 record={"frozen_before_formal_evaluation":True,"formal_eval_outputs_at_freeze":0,"training_updates":0,
  "base":BASE,"recorded_unix":time.time(),"sha256":{str(p):sha(p) for p in sorted(set(files))},
  "protocol_sha256":sha(ROOT/"protocol_plan.md"),"smoke":"runtime_smoke_audit.json"}
 assert not (ROOT/"frozen_sources.json").exists(),"Inspect any prior frozen study; do not overwrite silently"
 (ROOT/"frozen_sources.json").write_text(json.dumps(record,indent=2));print("ISSUE13_SOURCES_FROZEN",len(record["sha256"]),flush=True)
if __name__=="__main__":main()
