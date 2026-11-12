from pathlib import Path
p=Path('scripts/locomotion/analyze_issue16.py');s=p.read_text()
s=s.replace('import csv,gzip,hashlib,itertools,json,os','import csv,gzip,hashlib,itertools,json,os,subprocess')
s=s.replace(" candidates=[];original=json.loads((A/'cohort.json').read_text())"," candidates=[];original=json.loads((A/'cohort.json').read_text());selection_inputs={}\n manifest=json.loads((A/'sha256.json').read_text())\n assert (A/'sha256.json').read_bytes()==subprocess.check_output(['git','show','770014ee9a4393da6dff34eced89a04d83e788dd:'+str((A/'sha256.json').relative_to(REPO))])")
s=s.replace("   folder=Path(d['issue15_tracking']['output'])","   folder=A/'clean/moments'/f\"{m['evaluation_model']}_eval{es}\"\n   for source in [p,folder/'forward_05_yaw.npz',folder/'forward_05.npz']:\n    h=sha(source);assert h==manifest[str(source.relative_to(A))],str(source)\n    selection_inputs[str(source)]=h")
s=s.replace("best.update(command='forward_05'","best.update(selection_inputs_sha256=selection_inputs,committed_archive_manifest_sha256=sha(A/'sha256.json'),command='forward_05'")
compile(s,str(p),'exec');p.write_text(s);print('HEADING_SELECTION_BOUND_TO_COMMITTED_ARCHIVE')
