"""Summarize unchanged development gates; never promote a policy or certify control."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();r=a.root;evidence=[]
def read(name):
 path=r/name;d=json.loads(path.read_text())
 evidence.append(dict(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 return d
integrity={arm:read(arm+'/integrity.json') for arm in ['control','candidate']}
for arm,d in integrity.items():
 assert d['passed']
 path=r/arm/'fit/world_model_calibrated.pt'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==d['artifact_sha256']
branch=read('branch_validation.json')
for arm in ['control','candidate']:assert branch['model_sha256'][arm]==integrity[arm]['artifact_sha256']
gates={'initial_parent':read('candidate/initial_h8_gate.json'),'matched_control':read('paired_trajectory_gate.json'),'held_branch':branch}
rows=[]
for label,d in gates.items():
 checks=d['checks'];assert checks and all(isinstance(x['passed'],bool) for x in checks)
 assert d['passed']==all(x['passed'] for x in checks)
 rows.append(dict(gate=label,checks=len(checks),failed=sum(not x['passed'] for x in checks),passed=d['passed'],failures=[x for x in checks if not x['passed']]))
passed=all(x['passed'] for x in rows)
out=dict(development_gates_passed=passed,policy_promoted=False,final_goal_completed=False,
 decision='pending_physical_control_and_multiseed_acceptance' if passed else 'rejected_development_gates',
 candidate_sha256=integrity['candidate']['artifact_sha256'],gates=rows,evidence=evidence,
 scope='Final fixed-budget development gates only. No threshold changes, independent-motion success, multi-seed proof or promotion. Native response audits remain separate evidence.')
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['gates','evidence']}))
print(json.dumps([{k:v for k,v in row.items() if k!='failures'} for row in rows]))
