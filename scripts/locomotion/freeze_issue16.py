"""Freeze the final issue16 operator, physical sources, models and prospective inputs."""
import json,os,subprocess
from pathlib import Path
from issue16_common import ROOT,REPO,sha
from run_issue16 import build_jobs
def main():
 os.chdir(REPO);assert not (ROOT/'frozen_evaluation_sources.json').exists()
 smoke=json.loads((ROOT/'smoke_audit.json').read_text());assert smoke['passed']
 log=(ROOT/'mirror_unit_tests.log').read_text();assert 'Ran 9 tests' in log and log.rstrip().endswith('OK')
 d=json.loads((ROOT/'progress.json').read_text());p=Path('/proc')/str(d['pid']);assert not p.exists() or (p/'stat').read_text().split()[2]=='Z','Preflight child still live'
 c=json.loads((ROOT/'cohort.json').read_text());jobs=build_jobs(c);(ROOT/'formal_jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
 tracked=subprocess.check_output(['git','ls-files','source/unitree_rl_lab','scripts/locomotion','tests/locomotion']).decode().splitlines()
 paths={REPO/n for n in tracked if Path(n).suffix in ['.py','.sh','.json','.yaml','.yml','.urdf','.xml']}
 paths.update(REPO/p for p in Path('scripts/locomotion').glob('*issue16*.py'))
 paths.update(REPO/p for p in Path('tests/locomotion').glob('*issue16*.py'))
 paths.add(REPO/'tests/locomotion/fixtures/issue16_joint_axes.json');paths.add(REPO/'docs/stage1_issue16_20261111.md')
 for n in ['issue_source.json','cohort.json','protocol_plan.md','operator_definition_amendment.json','mirror_unit_tests.log','smoke_audit.json','usd_inspection.json','usd_collision_geometry.npz','formal_jobs.json','lineage_heading_gate.json']:
  paths.add(ROOT/n)
 inspection=json.loads((ROOT/'usd_inspection.json').read_text());fixture=json.loads((REPO/'tests/locomotion/fixtures/issue16_joint_axes.json').read_text())
 assert fixture['inspection_source_sha256']==inspection['source_sha256']==sha(REPO/'scripts/locomotion/inspect_issue16_usd.py')
 assert fixture['inspection_output_sha256']==sha(ROOT/'usd_inspection.json')
 assert fixture['usd_sha256']==inspection['usd_sha256']==sha(inspection['usd'])
 paths.add(Path(inspection['usd']))
 for layer in inspection['layers']:
  if layer['sha256']:
   assert sha(layer['path'])==layer['sha256'];paths.add(Path(layer['path']))
 for m in c['models'].values():assert sha(m['checkpoint'])==m['sha256'];paths.add(Path(m['checkpoint']))
 for n in ['cohort.json','node_statistics.json','seed_statistics.json','job_statistics.json']:
  paths.add(Path(c['baseline_archive'])/n)
 result=dict(base_commit=c['base_commit'],training_updates=0,formal_jobs=34,formal_initial_trials=203776,files={str(p.resolve()):sha(p) for p in sorted(paths)})
 (ROOT/'frozen_evaluation_sources.json').write_text(json.dumps(result,indent=2)+'\n')
 print('ISSUE16_FROZEN',len(paths),flush=True)
if __name__=='__main__':main()
