"""Restore and verify exact issue20 source bytes on the remote Linux server."""
import argparse,json,shutil
from pathlib import Path
from issue20_archive_codec import restore,contained,digest
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();archive=a.archive.resolve();out=a.output.resolve();assert out!=archive and not out.is_relative_to(archive);manifest=json.loads((archive/'source_files_manifest.json').read_text());chunks=json.loads((archive/'lossless_chunk_manifest.json').read_text())
for rel,row in manifest.items():
 dest=contained(out,rel)
 if rel in chunks:restore(archive,chunks[rel],out)
 else:
  source=contained(archive,rel);assert source.stat().st_size==row['bytes'] and digest(source)==row['sha256'];dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(source,dest)
 assert dest.stat().st_size==row['bytes'] and digest(dest)==row['sha256'],rel
print('ISSUE20_SOURCE_RESTORED_AND_SHA256_VERIFIED',len(manifest),str(out))
