"""Lossless remote archive; reuse committed originals and existing checkpoints."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
DEST=REPO/'validation_artifacts/stage1_issue18_20261113'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 os.chdir(REPO)
 assert read(ROOT/'evaluation_progress.json')['status']=='completed'
 assert read(ROOT/'postprocess_progress.json')['status']=='analysis_and_render_complete_manual_qa_report_archive_publication_pending'
 for name in ['preflight_audit.json','projection_validation.json','raw_analysis_audit.json','video_quality_audit.json']:assert read(ROOT/name)['passed']
 review=read(ROOT/'manual_visual_review.json');assert review['actual_review_completed']
 assert review['figures_reviewed']==3 and review['contact_sheets_reviewed']==1 and review['full_resolution_frames_reviewed']>=4
 for p,h in review['reviewed_files_sha256'].items():assert sha(ROOT/p)==h
 for section in ['sources','models','protocol']:
  for p,h in read(ROOT/'formal_inputs_frozen.json')[section].items():assert sha(p)==h,p
 for p,h in read(ROOT/'raw_analysis_audit.json')['raw_inputs_sha256'].items():assert sha(p)==h,p
 assert (ROOT/'results.md').exists() and (ROOT/'seed_tables.md').exists()
 assert read(ROOT/'original_reuse_portability.json')['all_original_bytes_already_committed']
 # Resolve existing archived checkpoints by scientific SHA; no new weights or copies.
 wanted={m['checkpoint_sha256'] for m in read(ROOT/'cohort.json')['models']};found={}
 tracked=subprocess.check_output(['git','ls-files','-z','validation_artifacts']).decode().split('\0')
 for name in tracked:
  if not name.endswith('.pt'):continue
  p=REPO/name;h=sha(p)
  if h in wanted:found.setdefault(h,name)
 assert wanted<=found.keys(),('checkpoints missing from existing committed archives',wanted-found.keys())
 refs=[]
 for m in read(ROOT/'cohort.json')['models']:
  refs.append(dict(model=m['model'],sha256=m['checkpoint_sha256'],existing_committed_checkpoint=found[m['checkpoint_sha256']],new_training=False))
 (ROOT/'checkpoint_reuse_references.json').write_text(json.dumps(refs,indent=2)+'\n')
 assert not DEST.exists(),'Existing archive requires audit before any modification'
 DEST.mkdir(parents=True);original={};chunks={}
 skip={'archive_receipt.json','publication_audit.json','push_receipt.json','github_issue_reply_receipt.json','notion_delivery_receipt.json'}
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.name in skip:continue
  assert not p.is_symlink();rel=p.relative_to(ROOT);h=sha(p);size=p.stat().st_size
  original[str(rel)]={'sha256':h,'bytes':size}
  if size>32*1024*1024:
   folder=DEST/'lossless_chunks'/rel;folder.mkdir(parents=True);parts=[];joined=hashlib.sha256()
   with p.open('rb') as reader:
    while data:=reader.read(8*1024*1024):
     out=folder/(f'{len(parts):04d}.part');out.write_bytes(data);joined.update(data)
     parts.append(dict(path=str(out.relative_to(DEST)),sha256=sha(out),bytes=len(data)))
   assert joined.hexdigest()==h
   chunks[str(rel)]=dict(original_sha256=h,original_bytes=size,parts=parts)
  else:
   out=DEST/rel;out.parent.mkdir(parents=True,exist_ok=True);os.link(p,out);assert sha(out)==h
  assert sha(p)==h
 (DEST/'.gitattributes').write_text('* -text\n')
 (DEST/'source_files_manifest.json').write_text(json.dumps(original,indent=2)+'\n')
 (DEST/'lossless_chunk_manifest.json').write_text(json.dumps(chunks,indent=2)+'\n')
 text='# Issue18 lossless remote archive\n\nStart with results.md and seed_tables.md. Zero training; 45 new strict projection jobs plus 45 reused originals. Original raw evidence and all 15 existing checkpoints are already committed: original_reuse_portability.json and checkpoint_reuse_references.json identify exact paths and SHA256 hashes. New formal results, moments, phase records, per-trial tables, figures, videos and validation evidence remain lossless. No default promotion.\n\nAbsolute provenance paths refer to the original server; archived relative suffixes resolve equivalent bytes. For a chunked file, concatenate part paths in manifest order and verify original size/SHA. Git text normalization is disabled. Publication receipts remain outside this commit to avoid circular self hashes.\n'
 (DEST/'ARCHIVE.md').write_text(text)
 manifest={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
 (DEST/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (ROOT/'archive_receipt.json').write_text(json.dumps(dict(archive=str(DEST),new_checkpoints=0,reused_checkpoints=15,source_files=len(original),archive_files=len(manifest)+1,manifest_sha256=sha(DEST/'sha256.json'),all_source_bytes_unchanged=True),indent=2)+'\n')
 print('ISSUE18_ARCHIVE_PASS',len(manifest)+1,flush=True)
if __name__=='__main__':main()
