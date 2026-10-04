#!/usr/bin/env python3
"""Full scoped corpus, user-reviewed near-limit policy, resumable Isaac FK shards."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path('/root/gpufree-data/datasets/practice9/tw75_dataset_v1')
OUT=Path('/root/gpufree-data/datasets/practice9/tw123_v1/processed')
POLICY='tw75_user_review_20261003_near_limits_advisory'
ADVISORY={'joint_near_limit_fraction','joint_near_limit_run'}

def eligible(record):
    quality=record.get('quality',{})
    reasons=set(quality.get('reasons',[]))
    return (record.get('quality_pass') is True or (bool(reasons) and reasons<=ADVISORY)) and record.get('clipped_fraction',1)==0

def standing_upright(row):
    if row['category']!='standing_upper':return True
    a=np.load(row['source']);angles=[]
    for hip,knee,ankle in [(1,4,7),(2,5,8)]:
        u=a[:,knee]-a[:,hip];v=a[:,ankle]-a[:,knee]
        cosine=(u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)).clip(1e-8)
        angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    return float(np.percentile(angles,90))<=40

def main(worker):
    OUT.mkdir(parents=True,exist_ok=True);status=OUT/f'prepare_{worker}.status';status.write_text('running')
    rows=json.load(open(ROOT/'selection.json'));kept=[r for r in rows if standing_upright(r)]
    rejected=[r['id'] for r in rows if r not in kept]
    (OUT/f'selection_{worker}.json').write_text(json.dumps({'kept':len(kept),'excluded_non_upright_ids':rejected}))
    batches=[kept[i:i+500] for i in range(0,len(kept),500)]
    for i,records in enumerate(batches):
        if i%2!=worker:continue
        dest=OUT/'batches'/f'{i:03d}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'manifest.json').exists():continue
        inp=dest/'inputs.txt';inp.write_text(''.join(r['source']+'\n' for r in records))
        raw=dest/'retarget_manifest.json'
        if not raw.exists():
            with open(dest/'retarget.log','w') as log:
                subprocess.run([sys.executable,'scripts/practice9/retarget_humanml3d.py','--input',str(inp),'--output-dir',str(dest/'retargeted'),'--retarget-root','/root/gpufree-data/datasets/practice9/tw123_v1/native_training_geometry','--allow-quality-failures','--continue-on-error'],stdout=log,stderr=subprocess.STDOUT,check=True)
        payload=json.load(open(raw));accepted=[];rejections=[]
        for record in payload['motions']:
            if eligible(record):
                r=dict(record);r['strict_quality_pass']=record['quality_pass'];r['quality_pass']=True;r['acceptance_policy']=POLICY;accepted.append(r)
            else:rejections.append({'id':record['id'],'quality':record['quality']})
        payload['motions']=accepted;payload['acceptance_policy']=POLICY
        (dest/'training_eligible.json').write_text(json.dumps(payload,indent=2));(dest/'rejected.json').write_text(json.dumps(rejections,indent=2))
        if not accepted:raise ValueError(f'No eligible records in batch {i}')
        with open(dest/'fk.log','w') as log:
            subprocess.run([sys.executable,'scripts/practice9/custom_motion_to_npz.py','--headless','--input-manifest',str(dest/'training_eligible.json'),'--output-dir',str(dest/'npz'),'--output-manifest',str(dest/'manifest.json'),'--fk-batch-size','128','--continue-on-error'],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('completed',i,'retarget eligible',len(accepted),flush=True)
    status.write_text('completed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[0,1],required=True);a=p.parse_args()
    try:main(a.worker)
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/f'prepare_{a.worker}.status').write_text('failed');raise
