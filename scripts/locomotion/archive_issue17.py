"""Remote lossless issue17 archive; preserve every scientific byte and final model."""
import hashlib,json,os
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113');REPO=Path('/root/gpufree-data/projects/HUMANOID_TW');DEST=REPO/'validation_artifacts/stage1_issue17_20261113'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for name in ['formal_training_complete','formal_evaluation_complete','statistics_complete','automatic_media_complete','figures_complete']:assert (ROOT/name).exists(),name
 for name in ['formal_training_audit.json','raw_analysis_audit.json','video_quality_audit.json']:assert json.loads((ROOT/name).read_text())['passed']
 review=json.loads((ROOT/'manual_review.json').read_text());assert review['actual_review_completed'] and review['figures_reviewed']==6 and review['contact_sheets_reviewed']==1 and review['full_resolution_frames_reviewed']==4
 for p,h in review['reviewed_files_sha256'].items():assert sha(ROOT/p)==h,p
 for name in ['frozen_training_sources.json','frozen_evaluation_sources.json']:
  for p,h in json.loads((ROOT/name).read_text())['files'].items():assert sha(p)==h,p
 conclusion=json.loads((ROOT/'scientific_conclusion.json').read_text());assert conclusion['summary'] and conclusion['next_recommendation'];assert (ROOT/'results.md').is_file()
 records=json.loads((ROOT/'training_records.json').read_text());assert len(records)==10 and sum(r['updates'] for r in records)==15000
 models=ROOT/'models';models.mkdir(exist_ok=True)
 for r in records:
  source=Path(r['checkpoint']);assert sha(source)==r['sha256'];out=models/(r['group']+'_s'+str(r['training_seed'])+'.pt')
  if not out.exists():os.link(source,out)
  assert sha(out)==r['sha256']
 assert not DEST.exists(),'Existing archive requires exact-byte audit before changes';DEST.mkdir(parents=True)
 original={};chunked={};skip={'publication_audit.json','push_receipt.json','github_issue_reply_receipt.json','notion_delivery_receipt.json'}
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.name in skip:continue
  assert not p.is_symlink();rel=p.relative_to(ROOT);h=sha(p);original[str(rel)]=dict(sha256=h,bytes=p.stat().st_size)
  if p.stat().st_size>32*1024*1024:
   folder=DEST/'lossless_chunks'/rel;folder.mkdir(parents=True);pieces=[];joined=hashlib.sha256()
   with p.open('rb') as reader:
    while data:=reader.read(8*1024*1024):
     target=folder/(f'{len(pieces):04d}.part');target.write_bytes(data);joined.update(data);pieces.append(dict(path=str(target.relative_to(DEST)),sha256=sha(target),bytes=len(data)))
   assert joined.hexdigest()==h;chunked[str(rel)]=dict(original_sha256=h,original_bytes=p.stat().st_size,parts=pieces)
  else:
   target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target);assert sha(target)==h
  assert sha(p)==h
 (DEST/'.gitattributes').write_text('* -text\n')
 (DEST/'source_files_manifest.json').write_text(json.dumps(original,indent=2)+'\n');(DEST/'lossless_chunk_manifest.json').write_text(json.dumps(chunked,indent=2)+'\n')
 (DEST/'ARCHIVE.md').write_text('# Issue17 lossless remote archive\n\nStart with results.md. Ten final model/optimizer checkpoints are in models/. All raw evaluation and validation evidence is preserved, including initial failed floating-clock and incomplete-recorder preflights, which are excluded from formal denominators. Formal training is10x1500updates; evaluation90jobs450records460800initialtrials.\n\nScientific bytes are unchanged, including raw logs,CSV andSVG. .gitattributes disables Git text normalization; stage this directory with git add -f and verify every committed blob. Large files use lossless_chunks: concatenate part paths in manifest order and verify original SHA256/size. Absolute provenance paths refer to the original server; archived relative suffixes resolve equivalent bytes. Publication receipts live outside their own commit to avoid circular hashes.\n')
 manifest={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()};(DEST/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (ROOT/'archive_receipt.json').write_text(json.dumps(dict(archive=str(DEST),models=10,source_files=len(original),archive_files=len(manifest)+1,manifest_sha256=sha(DEST/'sha256.json'),all_source_bytes_unchanged=True),indent=2)+'\n');print('ISSUE17_LOSSLESS_ARCHIVE_COMPLETE',len(manifest)+1,flush=True)
if __name__=='__main__':main()
