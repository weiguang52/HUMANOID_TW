"""Bind unchanged policy/strict-USD/baseline inputs to existing exact Git objects."""
import json,subprocess
from pathlib import Path
from issue21_common import ROOT,OLD,REPO,sha,save
BASE='b8ddc926a017d479869dfcc9183d15c653a5a4d7';arc='validation_artifacts/stage1_issue20_20261114/'
def oid(path):return subprocess.check_output(['git','rev-parse',BASE+':'+path],text=True).strip()
def main():
 previous=json.loads((OLD/'existing_reference_bindings.json').read_text());assert previous['passed'];save('existing_reference_bindings.json',previous)
 chunk=json.loads((REPO/arc/'lossless_chunk_manifest.json').read_text());files=['strict_symmetric.usd','colored_asset_integrity.json','joint_colors.json','actor_phase_validation.json','zero_phase_reproduction.json','runtime_strict_model_gate.json','physical_comparison_audit.json','left_authority_audit.json']+['evaluation/symmetric/moments/symmetric_s%d_eval1_k0/forward_05_yaw.npz'%ts for ts in range(1,6)]+['evaluation/symmetric/moments/symmetric_s%d_eval1_k0/%s_initial_state.npz'%(ts,command) for ts in range(1,6) for command in ['stand','forward_05']];rows=[]
 for relative in files:
  p=OLD/relative;row={'source':str(p),'source_sha256':sha(p),'bytes':p.stat().st_size,'base_commit':BASE,'base_archive_relative':relative}
  if relative in chunk:
   parts=chunk[relative]['parts'];row['lossless_manifest_path']=arc+'lossless_chunk_manifest.json';row['lossless_manifest_blob']=oid(row['lossless_manifest_path']);row['parts']=[{'path':arc+x['path'],'blob':oid(arc+x['path']),'sha256':x['sha256'],'bytes':x['bytes']} for x in parts]
  else:row['path']=arc+relative;row['blob']=oid(row['path']);assert (REPO/row['path']).read_bytes()==p.read_bytes()
  rows.append(row)
 save('existing_issue20_input_objects.json',{'passed':True,'base_commit':BASE,'inputs':rows,'weights_reused_unchanged':True,'no_new_training':True});print('ISSUE21_EXACT_EXISTING_INPUT_OBJECTS_BOUND',len(rows),flush=True)
if __name__=='__main__':main()
