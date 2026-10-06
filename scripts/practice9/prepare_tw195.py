import json,pathlib,hashlib,numpy as np
from tw75_sampling import balance
S=pathlib.Path('/root/gpufree-data/datasets/practice9/tw195_v1');OLD=S.parent/'tw154_v1'
audit=json.load(open(S/'final_audit.json'));assert len(audit)==1775
lookup={(r['split'],r['id']):r for r in audit}
def reasons(a):
 if 'error' in a:return ['solver_error']
 fail=[]
 for k,lim in [('stance_error_p95',.003),('flat_spread_p95',.005),('max_q_change',.8),('velocity_ratio_max',1.001),('upper_joint_change',1e-7),('joint_limit_violation',1e-6),('global_lift_m',.003),('max_root_z_change',.031)]:
  if a[k] is not None and a[k]>lim:fail.append(k)
 if a['new_leg_acc_max']>max(40.,1.5*a['old_leg_acc_max']):fail.append('acceleration')
 if a['min_sole']<-.0005:fail.append('penetration')
 return fail
summary={};decisions=[]
for split,file in [('train','train.json'),('validation','validation/manifest.json')]:
 m=json.load(open(OLD/file));chosen=[]
 for r in m['motions']:
  a=lookup[(split,r['id'])];why=reasons(a);diagnostic=split=='validation' and r.get('evaluation_class') in ['treadmill','nonplanar_review']
  decisions.append(dict(split=split,id=r['id'],reasons=why,diagnostic_original_preserved=diagnostic))
  if split=='train' and why:continue
  if split=='validation' and why and not diagnostic:
   if r['id']=='000346' and why==['velocity_ratio_max']:
    diagnostic=True;r=dict(r,evaluation_class='standing_upper_reference_limit_diagnostic');decisions[-1]['diagnostic_original_preserved']=True
   else:raise ValueError((r['id'],why))
  chosen.append(dict(r,contact_ik_applied=not diagnostic,original_file=r['file']))
 if split=='train':
  chosen=balance(chosen);counts={c:sum(r['category']==c for r in chosen) for c in ['walking','standing_upper']};assert counts['walking']>=300 and counts['standing_upper']>=1000,counts
  summary['train_counts']=counts
 # Subset sidecars identically for both groups; preserve every selected bit and frame.
 spec=m['contact_reference'];z=np.load(spec['file']);oldids=z['clip_ids'].tolist();off=np.r_[0,np.cumsum(z['lengths'])];ix=[oldids.index(r['id']) for r in chosen];out={k:z[k].copy() for k in z.files}
 for k in ['contact','known','source_contact4']:out[k]=np.concatenate([z[k][off[i]:off[i+1]] for i in ix])
 out['clip_ids']=z['clip_ids'][ix];out['lengths']=z['lengths'][ix];cp=S/f'{split}.contacts.npz';np.savez_compressed(cp,**out)
 for v in ['original','corrected']:
  rows=[]
  for r in chosen:
   new=dict(r)
   if v=='corrected' and r['contact_ik_applied']:new['file']=str(S/'corrected_final'/split/(r['id']+'.npz'));new['contact_ik_audit']=lookup[(split,r['id'])];new['sole_min_z']=new['contact_ik_audit']['min_sole']
   rows.append(new)
  mnew=dict(m,motions=rows,contact_reference=dict(spec,file=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest()),contact_ik=dict(variant=v,source='TW195'))
  if split=='train':dest=S/f'train_{v}.json'
  else:
   dest=S/'validation'/v/'manifest.json';dest.parent.mkdir(parents=True,exist_ok=True)
   for seed in [42,123,2026]:
    jobs=[dict(motion_id=i,steps=r['frames']+1,output=str(dest.parent/f'{r["id"]}.seed{seed}.json')) for i,r in enumerate(rows)];(dest.parent/f'jobs{seed}.json').write_text(json.dumps(jobs,indent=2))
   (S/'eval'/v).mkdir(parents=True,exist_ok=True)
   for r in rows:
    # Reuse original per-clip sidecar; no reindexing or time edits.
    one=json.load(open(OLD/'contact/eval'/f'{r["id"]}.json'));one['motions']=[r];(S/'eval'/v/f'{r["id"]}.json').write_text(json.dumps(one,indent=2))
  dest.write_text(json.dumps(mnew,indent=2))
 summary[split+'_clips']=len(chosen);summary[split+'_frames']=sum(r['frames'] for r in chosen)
(S/'selection.json').write_text(json.dumps(decisions,indent=2));(S/'data_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
