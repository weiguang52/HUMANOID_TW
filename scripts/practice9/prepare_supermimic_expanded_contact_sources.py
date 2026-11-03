"""Freeze broader native-contact development sources without outcome selection."""
import argparse,json,hashlib,datetime,collections
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
r=a.root;diag=r/'supermimic_tw_20261031/longterm_diagnostics'
paths=dict(train=r/'training_speed_blend_20261026/train.json',historical_validation=diag/'calibration_v1/validation.json',development8=diag/'grounded_dynamic_cop_clarabel_init_v1/diagnostic_references_v1/collection_plan.json',final7=r/'independent_holdout_20261025/test.json')
inputs={k:json.loads(v.read_text()) for k,v in paths.items()};hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
assert hashes['train']=='4536b44359dc09a16b17b19de6c3a080c859aa85b96ed7b38a69f619deb9d054' and hashes['final7']=='aed912a2f82867d7b76aeb35f19a58f9595f1d2ea9240794260255d6bfda7541'
excluded={k:{x['source_family'] for x in inputs[k]['motions']} for k in ['historical_validation','development8','final7']};blocked=set().union(*excluded.values())
by_family={}
for x in sorted(inputs['train']['motions'],key=lambda x:x['id']):
 if x['source_family'] not in blocked:by_family.setdefault(x['source_family'],x)
quotas={'KIT':18,'CMU':4,'EKUT':1,'ACCAD':1};salt='tw-expanded-native-contact-v1:';selected=[];availability={}
for corpus,quota in quotas.items():
 candidates=[x for x in by_family.values() if x['source_family'].split('/')[2]==corpus]
 candidates.sort(key=lambda x:hashlib.sha256((salt+x['source_family']).encode()).hexdigest());assert len(candidates)>=quota
 availability[corpus]=len(candidates)
 for x in candidates[:quota]:
  source=Path(x['file']);assert source.is_file()
  selected.append(dict(id=x['id'],source_family=x['source_family'],corpus=corpus,file=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),frames=x['frames'],category=x['category'],strict_retarget_quality_pass=x.get('strict_retarget_quality_pass'),split='fit',sampling_rank=hashlib.sha256((salt+x['source_family']).encode()).hexdigest()))
assert len(selected)==len({x['source_family'] for x in selected})==24
for families in excluded.values():assert not {x['source_family'] for x in selected}&families
out=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),contract='expanded_native_contact_sources_v1',inputs={k:dict(file=str(paths[k]),sha256=hashes[k]) for k in paths},training_motions=len(inputs['train']['motions']),training_families=len({x['source_family'] for x in inputs['train']['motions']}),excluded_family_counts={k:len(v) for k,v in excluded.items()},eligible_distinct_families=len(by_family),eligible_by_corpus=availability,quotas=quotas,selection_salt=salt,selection='Lowest salted SHA256 source-family ranks within fixed corpus quotas; deduplicate using lowest ID; no quality or physical outcomes used.',motions=selected,phase_fractions=[0,.25,.5,.75],max_steps_per_phase=100,seed=42,planned_episodes=96,action='Unchanged reference PD target inverse nominal offset/scale; stop on original termination, no sample replacement.',grounding_status='Original332 references only: expanded grounding/contact audits not yet run; no claim of dynamic feasibility.',validation_labels_loaded=False,policy_acceptance=False,training_export_approved=False,scope='Broader fit-source collection plan only; original historical/development/final validation sets are unchanged. Existing category walking can include mixed actions; captions must be reviewed before labeling capability.')
a.output.mkdir(parents=True);(a.output/'source_plan.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['training_motions','training_families','excluded_family_counts','eligible_distinct_families','eligible_by_corpus','quotas','planned_episodes']}))
