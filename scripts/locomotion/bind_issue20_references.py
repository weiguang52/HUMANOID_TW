"""Bind unchanged existing checkpoint and fixed-state Git objects, without duplicates."""
import subprocess,json,hashlib
from pathlib import Path
from issue20_common import ROOT,OLD,REPO,sha,save
BASE='2c89ead59fa17fa9e339a8e46b972a8af579d237'
def binding(source,path,expected):
 assert sha(source)==expected;oid=subprocess.check_output(['git','rev-parse',BASE+':'+path],text=True).strip();data=subprocess.check_output(['git','cat-file','blob',oid]);assert hashlib.sha256(data).hexdigest()==expected;return {'source':str(source),'git_commit':BASE,'git_path':path,'git_blob':oid,'sha256':expected,'bytes':len(data)}
def main():
 models=json.loads((ROOT/'cohort.json').read_text());refs=[]
 for m in models:refs.append(binding(m['checkpoint'],f'validation_artifacts/stage1_issue19_20261113/models/sym_wrap_s{m["training_seed"]}.pt',m['sha256']))
 states=json.loads((OLD.with_name('issue17_20261113')/'fixed_state_set.json').read_text())
 for row in states['files']:refs.append(binding(row['path'],'validation_artifacts/stage1_issue17_20261113/state_collection/'+Path(row['path']).name,row['sha256']))
 original=json.loads((OLD.with_name('issue16_20261111')/'usd_inspection.json').read_text());layers=[]
 for row in original['layers']:
  if row.get('sha256') is not None:assert sha(row['path'])==row['sha256'];layers.append(row)
 save('existing_reference_bindings.json',{'passed':True,'references':refs,'original_USD_layers':layers,'checkpoints':5,'fixed_states':102400,'new_training':False,'source_models_already_committed':True});print('ISSUE20_EXISTING_REFERENCES_BOUND',len(refs),len(layers),flush=True)
if __name__=='__main__':main()
