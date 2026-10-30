from pathlib import Path
import json,subprocess,datetime
b=Path(__file__).parent;root=Path('/root/gpufree-data/projects/HUMANOID_TW');tests=[x['test'] for x in json.load(open('/root/twrl_runs/load_aligned_20261027/unit_suite_audit.json'))['results']]+['test_contact_history.py','test_lookback_sampling.py'];rows=[]
for t in tests:
 p=subprocess.run(['python',str(root/'scripts/practice9'/t)],cwd=root,capture_output=True,text=True);(b/(t+'.log')).write_text(p.stdout+p.stderr);rows.append({'test':t,'exit_code':p.returncode});print(t,p.returncode,flush=True)
(b/'unit_suite_audit.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_passed':all(x['exit_code']==0 for x in rows),'results':rows},indent=2))
