"""Freeze every source artifact losslessly, with already-committed model/state references."""
from pathlib import Path
import json,shutil,os,time
from issue20_common import ROOT,REPO,sha,save
from issue20_archive_codec import encode,decoded,contained
import hashlib

def main():
 assert json.loads((ROOT/'manual_visual_review.json').read_text())['actual_review_completed'];assert json.loads((ROOT/'prepublication_requirement_audit.json').read_text())['passed'];pub=ROOT/'publication';pub.mkdir(exist_ok=True);dest=REPO/'validation_artifacts/stage1_issue20_20261114';assert not dest.exists(),'Existing archive must be audited before any rebuild';dest.mkdir()
 cache=json.loads((pub/'precompression_manifest.json').read_text()) if (pub/'precompression_manifest.json').exists() else {'files':[]};cached={x['path']:x for x in cache['files']};chunks={};manifest={};sources=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and 'publication' not in p.relative_to(ROOT).parts and p.name not in ['archive.log']]
 for i,p in enumerate(sources):
  rel=str(p.relative_to(ROOT));digest=sha(p);manifest[rel]={'sha256':digest,'bytes':p.stat().st_size}
  if rel in cached:
   row=cached[rel];assert row['sha256']==digest and row['bytes']==p.stat().st_size
   for part in row['parts']:
    src=contained(pub/'codec_cache',part['path']);target=contained(dest,part['path']);target.parent.mkdir(parents=True,exist_ok=True);assert sha(src)==part['sha256'];shutil.copy2(src,target);assert sha(target)==part['sha256']
   h=hashlib.sha256();count=0
   with decoded(dest,row) as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b);count+=len(b)
   assert count==row['bytes'] and h.hexdigest()==digest;chunks[rel]=row
  else:
   row=encode(p,dest,rel,pub)
   if row is not None:chunks[rel]=row
  if i%100==0:print('ARCHIVE_FILE',i,'of',len(sources),rel,flush=True)
 for p in sources:
  row=manifest[str(p.relative_to(ROOT))];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
 (dest/'.gitattributes').write_text('* -text -filter -diff -merge\n');(dest/'source_files_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(dest/'lossless_chunk_manifest.json').write_text(json.dumps(chunks,indent=2)+'\n');(dest/'RESTORE.md').write_text('# Lossless source restoration\n\nAll small files retain their exact bytes. Most files larger than4MiB are stored in verified parts under .chunks/. MP4 deliveries up to90MiB retain their original paths for direct playback. Text/USD files are gzip-compressed with deterministic headers; arrays/videos are split without recompression. lossless_chunk_manifest.json binds original byte counts/SHA256 and every stored part.\n\nOn the remote Linux server, restore a complete source tree into a new directory with:\n\n    python scripts/locomotion/restore_issue20_archive.py --archive validation_artifacts/stage1_issue20_20261114 --output /root/tw_stage1/issue20_restored\n\nNo local copy is required. Existing_reference_bindings.json identifies all five unchanged final checkpoints and102400 fixed states by exact Git commit,path,blob and SHA256; those existing Git objects are not duplicated here. input_assets/original preserves the complete original USD layer structure; the strict USD is self-contained. Metadata keeps original absolute paths for provenance. Rebase /root/tw_stage1/issue20_20261114 to the restored root for inspection; do not alter hashed source files.\n')
 (dest/'lossless_archive_validation.json').write_text(json.dumps({'passed':True,'source_files':len(sources),'large_payloads_decoded_and_verified':len(chunks),'source_files_unchanged':True,'source_manifest_sha256':sha(dest/'source_files_manifest.json'),'chunk_manifest_sha256':sha(dest/'lossless_chunk_manifest.json'),'all_stored_part_digests_verified':True},indent=2)+'\n')
 hashes={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()};(dest/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n');assert len(hashes)+1==len([p for p in dest.rglob('*') if p.is_file()]);receipt={'passed':True,'source_files':len(sources),'archive_files':len(hashes)+1,'source_bytes':sum(x['bytes'] for x in manifest.values()),'archive_bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'source_files_unchanged':True,'all_large_payloads_decoded_and_SHA256_verified':True,'archive':str(dest),'time':time.time()};(pub/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('ISSUE20_LOSSLESS_ARCHIVE_PASS',receipt,flush=True)
if __name__=='__main__':main()
