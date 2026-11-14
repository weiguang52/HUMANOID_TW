"""Lossless issue19 archive, after actual visual review and all requirement audits."""
import hashlib,json,os
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
DEST=REPO/'validation_artifacts/stage1_issue19_20261113'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert (ROOT/'formal_training_complete').exists() and (ROOT/'formal_evaluation_complete').exists()
 for name in ['actor_pretraining_validation.json','preflight_validation.json','preflight_optimizer_validation.json','evaluation_preflight_validation.json','final_model_validation.json','raw_analysis_audit.json','video_quality_audit.json','prepublication_requirement_audit.json']:
  assert read(ROOT/name)['passed'],name
 review=read(ROOT/'manual_visual_review.json')
 assert review['actual_review_completed'] and review['figures_reviewed']==4 and review['contact_sheets_reviewed']==1 and review['full_resolution_frames_reviewed']>=9
 for p,h in review['reviewed_files_sha256'].items():assert sha(ROOT/p)==h,p
 for name in ['frozen_training_sources.json','frozen_evaluation_sources.json']:
  for p,h in read(ROOT/name)['files'].items():assert sha(p)==h,p
 for p,h in read(ROOT/'raw_analysis_audit.json')['raw_inputs_sha256'].items():assert sha(p)==h,p
 portable=read(ROOT/'reference_reuse_portability.json');assert portable['passed'] and portable['all_reference_bytes_already_committed']
 records=read(ROOT/'training_records.json');assert len(records)==10
 models=ROOT/'models';models.mkdir(exist_ok=True)
 for r in records:
  source=Path(r['checkpoint']);assert sha(source)==r['sha256']
  target=models/(r['group']+'_s'+str(r['training_seed'])+'.pt')
  if not target.exists():os.link(source,target)
  assert sha(target)==r['sha256']
 assert not DEST.exists(),'Audit a partial archive before any retry'
 DEST.mkdir(parents=True);original={};chunks={}
 skip={'archive.log','archive_receipt.json','publication_audit.json','push_receipt.json','github_issue_reply_receipt.json','notion_delivery_receipt.json'}
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.name in skip or p.is_relative_to(ROOT/'publication'):continue
  assert not p.is_symlink();rel=p.relative_to(ROOT);size=p.stat().st_size;h=sha(p)
  original[str(rel)]=dict(sha256=h,bytes=size)
  if size>32*1024*1024:
   folder=DEST/'lossless_chunks'/rel;folder.mkdir(parents=True);parts=[];joined=hashlib.sha256()
   with p.open('rb') as reader:
    while data:=reader.read(8*1024*1024):
     target=folder/(str(len(parts)).zfill(4)+'.part');target.write_bytes(data);joined.update(data)
     parts.append(dict(path=str(target.relative_to(DEST)),sha256=sha(target),bytes=len(data)))
   assert joined.hexdigest()==h
   chunks[str(rel)]=dict(original_sha256=h,original_bytes=size,parts=parts)
  else:
   target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
   assert sha(target)==h
  assert sha(p)==h
 (DEST/'.gitattributes').write_text('* -text\n')
 (DEST/'source_files_manifest.json').write_text(json.dumps(original,indent=2)+'\n')
 (DEST/'lossless_chunk_manifest.json').write_text(json.dumps(chunks,indent=2)+'\n')
 text='# Issue19 lossless remote archive\n\nStart with results.md and seed_tables.md. Ten final actor/critic/optimizer checkpoints are in models/. Training is exactly10x1500 additional updates with4096 environments. New formal evaluation is60 jobs,270 command records,276480 initial trials;45 historical reference jobs are reused without rerun. All105 audited jobs produce495 records and506880 initial trial rows. Smoke/preflight and media reconstruction do not enter formal denominators.\n\nThe sym_wrap actor is already structurally projected: do not add another inference wrapper. Reload its checkpoint with the issue19 actor installed before load_state_dict (eval_issue19.py), or deploy onnx/sym_wrap_sN.onnx directly (101 input,30 output). The standard continuation arm remains unwrapped.\n\nScientific outcome and any failed gate remain explicit. The single next recommendation is not implemented. References and five parent sym_int checkpoints are already in committed issue17/18 archives; reference_reuse_portability.json binds exact bytes and Git blobs.\n\nEvery raw result, phase/feedback record, trial table, startup/slope/half-cycle array, validation file, figure and colored video remains lossless. Absolute provenance paths identify the original server. For chunked files concatenate parts in manifest order and verify original size/SHA256. Git text conversion is disabled. Publication receipts are kept outside their own commit. Frozen scientific files must not be modified through their hardlinked source paths after archival.\n'
 (DEST/'ARCHIVE.md').write_text(text)
 manifest={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
 (DEST/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
 receipt=dict(archive=str(DEST),new_final_checkpoints=10,source_files=len(original),archive_files=len(manifest)+1,manifest_sha256=sha(DEST/'sha256.json'),all_source_bytes_unchanged=True)
 (ROOT/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('ISSUE19_LOSSLESS_ARCHIVE_PASS',receipt,flush=True)
if __name__=='__main__':main()
