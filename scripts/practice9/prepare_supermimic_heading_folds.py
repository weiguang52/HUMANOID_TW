"""Freeze corpus-stratified source-family folds before native response labels."""
import argparse,json,hashlib,collections
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
x=json.loads(a.manifest.read_text());assert x['contract']=='expanded_native_contact_sources_v1';motions=x['motions'];assert len(motions)==24 and len({m['source_family'] for m in motions})==24
folds=[[] for _ in range(4)];salt='tw-contact-response-folds-v1:'
for corpus in sorted({m['corpus'] for m in motions}):
 group=sorted([m for m in motions if m['corpus']==corpus],key=lambda m:hashlib.sha256((salt+m['source_family']).encode()).hexdigest())
 corpus_count=[0]*4
 for m in group:
  i=min(range(4),key=lambda i:(len(folds[i]),corpus_count[i],i));folds[i].append(m['source_family']);corpus_count[i]+=1
assert [len(f) for f in folds]==[6]*4 and len(set(sum(folds,[])))==24
out=dict(contract='source_family_heading_fourfold_v1',manifest=str(a.manifest),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),salt=salt,folds=[dict(name='fold'+str(i),held_out_families=sorted(f),train_families=sorted(set(sum(folds,[]))-set(f))) for i,f in enumerate(folds)],updates=1000,batch=128,lr=5e-4,seed=42,hidden=[128,128],response_only=True,selection='Corpus groups sorted; salted family hash order; assign to least-populated fold, then least corpus count, then fold index.',labels_loaded=False,policy_acceptance=False)
a.output.write_text(json.dumps(out,indent=2));print('FROZEN_FOLDS',[len(f) for f in folds],hashlib.sha256(a.output.read_bytes()).hexdigest())
