"""Build new split manifests and contact labels only from regenerated FK clips."""
import json,subprocess,sys
from pathlib import Path
from build_contact_references import build
S=Path('/root/gpufree-data/datasets/practice9/tw123_v1')
selection=Path('/root/gpufree-data/datasets/practice9/tw75_dataset_v1/selection.json')
for worker in (0,1):assert (S/f'processed/prepare_{worker}.status').read_text()=='completed'
assert (S/'pilot.status').read_text().strip()=='completed'
assert (S/'pilot_review.json').is_file(),'Visual/FK review not completed'
files=sorted((S/'processed/batches').glob('*/manifest.json'))
subprocess.run([sys.executable,'scripts/practice9/build_tw75_manifests.py',
    '--selection',str(selection),'--fk-manifests',*map(str,files),
    '--output',str(S/'processed/splits')],check=True)
all_rows={}
families={}
for split in ('train','val','test'):
    path=S/f'processed/splits/{split}.json';d=json.loads(path.read_text())
    families[split]={r['source_family'] for r in d['motions']}
    all_rows.update({r['id']:(r,d) for r in d['motions']})
    build(path,S/f'contact/splits/{split}.json')
assert not families['train'] & (families['val']|families['test'])
for clip in ('000016','000124','000346','000039','000139','000211'):
    row,template=all_rows[clip]
    assert row['split']=='val'
    d=dict(template,motions=[dict(row,weight=1.)])
    path=S/f'processed/eval/{clip}.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(d,indent=2))
    build(path,S/f'contact/eval/{clip}.json')
subprocess.run([sys.executable,'scripts/practice9/audit_tw123_data.py'],check=True)
subprocess.run([sys.executable,'scripts/practice9/prepare_tw123_validation.py'],check=True)
(S/'data.status').write_text('completed\n')
