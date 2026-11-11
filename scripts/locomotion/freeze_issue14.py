"""Freeze protocol, matched historical behavior and all formal training dependencies."""
import ast,difflib,hashlib,json,subprocess,time
from pathlib import Path
from issue14_common import ROOT,REPO,BASE,sha

class RemoveOptIn(ast.NodeTransformer):
 def __init__(self):self.flags=0;self.guards=0
 def visit_Expr(self,node):
  value=node.value
  if isinstance(value,ast.Call) and value.args and isinstance(value.args[0],ast.Constant) and value.args[0].value=="--critic_recovery_clock":
   self.flags+=1;return None
  return self.generic_visit(node)
 def visit_If(self,node):
  if ast.unparse(node.test)=="args_cli.critic_recovery_clock":
   self.guards+=1
   return [self.visit(x) for x in node.orelse]
  return self.generic_visit(node)

def main():
 assert (ROOT/"runtime_smoke_complete").exists()
 assert json.loads((ROOT/"runtime_smoke_audit.json").read_text())["passed"]
 assert json.loads((ROOT/"eval_adapter_smoke_audit.json").read_text())["passed"]
 assert "Ran 10 tests" in (ROOT/"unit_tests.log").read_text() and "\nOK\n" in (ROOT/"unit_tests.log").read_text()
 assert not (ROOT/"formal_training_state.json").exists()
 assert not list((REPO/"logs/rsl_rl/tw_stage1").glob("*_issue14_20261111_clock_s*"))
 parent=json.loads((ROOT/"input_parents.json").read_text())
 assert parent==json.loads((Path("/root/gpufree-data/tw_stage1/issue12_20261110")/"input_models.json").read_text())
 for m in parent.values():assert sha(m["checkpoint"])==m["sha256"]
 original=subprocess.check_output(["git","show",BASE+":scripts/locomotion/train.py"]).decode();current=Path("scripts/locomotion/train.py").read_text()
 transformer=RemoveOptIn();stripped=transformer.visit(ast.parse(current))
 assert transformer.flags==1 and transformer.guards==3
 old_ast=ast.dump(ast.parse(original),include_attributes=False);new_ast=ast.dump(stripped,include_attributes=False)
 assert old_ast==new_ast,"Default-off source has another semantic change"
 (ROOT/"train_opt_in_source_diff.patch").write_text("".join(difflib.unified_diff(original.splitlines(True),current.splitlines(True),fromfile="archived_issue12_train.py",tofile="train_with_opt_in_critic_clock.py")))
 (ROOT/"default_off_source_audit.json").write_text(json.dumps({"passed":True,"original_sha256":hashlib.sha256(original.encode()).hexdigest(),"current_sha256":sha("scripts/locomotion/train.py"),
 "same_AST_after_removing_exactly_one_cli_argument_and_three_opt_in_guards":True,"legacy_AST_sha256":hashlib.sha256(old_ast.encode()).hexdigest(),"stripped_AST_sha256":hashlib.sha256(new_ast.encode()).hexdigest()},indent=2))
 old=json.loads(Path("/root/gpufree-data/tw_stage1/issue12_20261110/frozen_training_sources.json").read_text())["sha256"];historical={}
 for name,h in old.items():
  if name.endswith("/scripts/locomotion/train.py") or name=="scripts/locomotion/train.py":
   assert hashlib.sha256(original.encode()).hexdigest()==h
   historical[name]={"identical_original":True,"current_default_off_AST_and_runtime_reproduced":True,"original_sha256":h}
  else:
   assert sha(name)==h,("Historical training dependency changed",name)
   historical[name]={"identical_original":True,"sha256":h}
 (ROOT/"historical_source_equivalence_audit.json").write_text(json.dumps({"passed":True,"original_frozen_files":len(historical),"checks":historical,"intervention":"Only the optional critic observation/input expansion; all original behavior dependencies byte-identical."},indent=2))
 files=set(Path(x) for x in old)
 files.update(Path("scripts/locomotion").glob("*issue14*.py"));files.add(Path("scripts/locomotion/train.py"))
 files.update(Path("source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw").glob("*.py"))
 files.update(Path(x) for x in ["tests/locomotion/test_issue14_clock.py","docs/stage1_issue14_20261111.md","scripts/locomotion/issue13_tracking.py","scripts/locomotion/issue13_statistics.py","scripts/locomotion/issue13_analysis_data.py","scripts/locomotion/eval_cmd.py","scripts/locomotion/failure_traces.py","scripts/locomotion/trial_risk_records.py"])
 for p in files:
  assert p.exists(),p
  if p.suffix==".py":ast.parse(p.read_text())
 identity={str(p):sha(p) for p in sorted(files,key=str)}
 output=ROOT/"frozen_training_sources.json";assert not output.exists()
 output.write_text(json.dumps({"base":BASE,"frozen_before_formal_training":True,"formal_runs_at_freeze":0,"frozen_unix":time.time(),"sha256":identity,
 "formal_models":5,"updates_each":1500,"num_envs":4096,"parent_models":parent,"default_off_proven":True,"historical_code_equivalence_proven":True},indent=2))
 assert (ROOT/"protocol_plan.md").read_bytes()==Path("docs/stage1_issue14_20261111.md").read_bytes()
 subprocess.run(["git","diff","--check"],check=True)
 (ROOT/"formal_gates_complete").write_text("10unit tests, strict parent/Adam migration, bitwise off PPO regression,on1000step rollover and two evaluator smokes; protocol and sources frozen before five formal runs.\n")
 print("ISSUE14_FORMAL_GATES_FROZEN",len(identity),flush=True)
if __name__=="__main__":main()
