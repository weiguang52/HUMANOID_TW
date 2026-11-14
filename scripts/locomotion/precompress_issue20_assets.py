"""Precompute lossless immutable asset payloads while formal evaluation runs."""
from pathlib import Path
from issue20_common import ROOT,sha
from issue20_archive_codec import encode
import json,time
pub=ROOT/'publication';pub.mkdir(exist_ok=True);cache=pub/'codec_cache';cache.mkdir(exist_ok=True);rows=[]
files=[ROOT/'usd_attribute_differences.csv',ROOT/'strict_symmetric.usd',*sorted((ROOT/'input_assets/original').rglob('*.usd'))]
for p in files:
 if p.stat().st_size<=4*1024*1024:continue
 before=sha(p);row=encode(p,cache,str(p.relative_to(ROOT)),pub);assert row['sha256']==before==sha(p);rows.append(row);print('IMMUTABLE_ASSET_LOSSLESS_CACHE',row['path'],row['bytes'],sum(x['bytes'] for x in row['parts']),len(row['parts']),flush=True)
(pub/'precompression_manifest.json').write_text(json.dumps({'passed':True,'files':rows,'source_files_unchanged':True,'time':time.time()},indent=2)+'\n');print('ISSUE20_IMMUTABLE_ASSET_CACHE_COMPLETE',flush=True)
