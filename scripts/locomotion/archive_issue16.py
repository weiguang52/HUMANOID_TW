"""Losslessly archive immutable issue16 evidence on the remote Linux filesystem."""
import hashlib,json,os
from pathlib import Path
from issue16_common import ROOT,REPO,sha
from run_issue16 import verify_frozen
DEST=REPO/'validation_artifacts/stage1_issue16_20261111'
CHUNK=8*1024*1024
def main():
 verify_frozen()
 for name in ['formal_evaluation_complete','initial_postprocessing_complete','automated_media_complete','derived_automatic_complete']:
  assert (ROOT/name).is_file(),name
 a=json.loads((ROOT/'analysis_integrity_audit.json').read_text())
 assert a['passed'] and a['formal_jobs']==34 and a['formal_command_records']==199 and a['initial_trials']==203776 and a['all_trial_heading_series_audited_commands']==133
 runtime=json.loads((ROOT/'runtime_symmetry_audit.json').read_text())
 assert runtime['audit_completed']
 review=json.loads((ROOT/'manual_review.json').read_text())
 assert review['actual_review_completed'] and review['figures_reviewed']==7 and review['contact_sheets_reviewed']==1 and review['full_resolution_frames_reviewed']==6
 conclusion=json.loads((ROOT/'scientific_conclusion.json').read_text())
 assert conclusion['overall_classification'] in ['policy_bias','robot_sim_asymmetry','both','unable_to_distinguish']
 assert len(conclusion['next_recommendations'])==1 and not conclusion['recommendation_implemented']
 assert not conclusion['production_default_promoted'] and conclusion['training_updates']==0
 (ROOT/'results.md').read_text()
 assert not DEST.exists(),'Archive already exists: verify existing bytes instead of overwriting'
 DEST.mkdir(parents=True)
 raw_manifest={};chunked={}
 excluded={'publication_audit.json','publication_receipt.json','github_issue_reply_receipt.json','notion_reply_receipt.json','api_push_receipt.json'}
 for source in sorted(ROOT.rglob('*')):
  if not source.is_file() or source.name in excluded:continue
  rel=source.relative_to(ROOT)
  assert not source.is_symlink(),str(source)
  before=sha(source);raw_manifest[str(rel)]=dict(sha256=before,bytes=source.stat().st_size)
  if source.stat().st_size>32*1024*1024:
   target=DEST/'lossless_chunks'/rel;target.mkdir(parents=True)
   parts=[];joined=hashlib.sha256()
   with source.open('rb') as reader:
    while data:=reader.read(CHUNK):
     name=f'{len(parts):04d}.part';out=target/name;out.write_bytes(data);joined.update(data)
     parts.append(dict(path=str(out.relative_to(DEST)),bytes=len(data),sha256=sha(out)))
   assert joined.hexdigest()==before
   chunked[str(rel)]=dict(original_bytes=source.stat().st_size,original_sha256=before,parts=parts,
    reconstruction='Concatenate listed part paths in the listed order as raw bytes. No compression or text conversion.')
  else:
   out=DEST/rel;out.parent.mkdir(parents=True,exist_ok=True);os.link(source,out)
   assert sha(out)==before
  assert sha(source)==before
 (DEST/'source_files_manifest.json').write_text(json.dumps(raw_manifest,indent=2)+'\n')
 (DEST/'lossless_chunk_manifest.json').write_text(json.dumps(chunked,indent=2)+'\n')
 (DEST/'ARCHIVE.md').write_text('''# Issue16 evaluation and heading-source analysis

Remote Linux evidence from issue16_20261111. Start with results.md. This archive preserves exact source bytes, including preflight corrections and raw logs. It contains no new training weights: cohort.json binds the existing eleven immutable checkpoints and their SHA-256 digests.

Large files are stored losslessly in lossless_chunks/. Reconstruct each original file by concatenating its part paths in the order listed in lossless_chunk_manifest.json; verify the resulting byte count and original SHA-256. The original frozen source files remain unchanged on the server. All other source_files_manifest.json entries retain their original relative paths and exact bytes.

Preflight_phase_definition_superseded is forensic validation evidence, excluded from formal trial denominators. Formal denominators are 34 jobs,199 command records,203776 initial trials. Original lineage and the representative trial selection reuse committed issue15 evidence. Runtime readback and replay reconstruction are separate validations, not additional formal trials.

Frozen absolute paths in provenance describe the original remote experiment. Resolve archived equivalents by relative suffix under this directory, using lossless chunk reconstruction where listed. Source shaders/USD/checkpoints already bound by frozen provenance remain remote source assets rather than being copied to the local client.
''')
 manifest={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
 (DEST/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for rel,item in raw_manifest.items():
  if rel in chunked:continue
  assert sha(DEST/rel)==item['sha256']
 verify_frozen()
 receipt=dict(archive=str(DEST),source_files=len(raw_manifest),archive_files=len(manifest)+1,
  chunked_files=len(chunked),all_source_bytes_unchanged=True,manifest_sha256=sha(DEST/'sha256.json'),source_sha256=sha(__file__))
 (ROOT/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
