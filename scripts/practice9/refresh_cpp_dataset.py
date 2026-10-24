"""Versioned cpp refresh: regenerate the existing training selection, never mix solvers."""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path
STATE=Path('/root/gpufree-data/datasets/practice9/cpp_refresh_20261024')
DATA=STATE.parent
SOLVER=STATE/'native_training_geometry'
SOURCE=DATA/'tw215_v1/train.json'
HUMAN=DATA/'humanml3d_rebuild/staging-v1/HumanML3D/new_joints'

def run(cmd, log):
    with log.open('w') as stream:
        subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)

def retarget_batch(item):
    index,rows=item; dest=STATE/'batches'/f'{index:03d}';dest.mkdir(parents=True,exist_ok=True)
    inp=dest/'inputs.txt';inp.write_text(''.join(str(HUMAN/(r['id']+'.npy'))+'\n' for r in rows))
    run([sys.executable,'scripts/practice9/retarget_humanml3d.py','--input',str(inp),'--output-dir',str(dest/'retargeted'),'--retarget-root',str(SOLVER),'--allow-quality-failures','--continue-on-error'],dest/'retarget.log')
    return json.loads((dest/'retarget_manifest.json').read_text())

def retarget():
    from prepare_tw123_training import eligible, POLICY
    original=json.loads(SOURCE.read_text());rows=original['motions'];lookup={r['id']:r for r in rows}
    batches=list(enumerate([rows[i:i+150] for i in range(0,len(rows),150)]))
    accepted=[];rejected=[];template=None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for payload in pool.map(retarget_batch,batches):
            template=payload
            for r in payload['motions']:
                if eligible(r):
                    accepted.append(dict(r,quality_pass=True,strict_quality_pass=r['quality_pass'],acceptance_policy=POLICY))
                else:rejected.append(dict(id=r['id'],quality=r['quality']))
            rejected.extend(payload['failures'])
            print('retarget accepted',len(accepted),'rejected',len(rejected),flush=True)
    template=dict(template,motions=accepted,failures=rejected)
    assert accepted,'No eligible regenerated clips'
    (STATE/'retarget_manifest.json').write_text(json.dumps(template,indent=2))
    (STATE/'source_manifest.json').write_text(json.dumps(original,indent=2))
    (STATE/'retarget_rejections.json').write_text(json.dumps(rejected,indent=2))

def fk():
    run([sys.executable,'scripts/practice9/custom_motion_to_npz.py','--headless','--input-manifest',str(STATE/'retarget_manifest.json'),'--output-dir',str(STATE/'npz'),'--output-manifest',str(STATE/'fk_manifest.json'),'--fk-batch-size','128','--continue-on-error'],STATE/'fk.log')

def correct(row):
    from contact_ik import process
    import smooth_tw195
    smooth_tw195.S=STATE
    try:
        raw=STATE/'corrected/train'/(row['id']+'.npz')
        process(row,raw)
        return smooth_tw195.job(('train',row))
    except Exception as error:
        return dict(id=row['id'],error=str(error))

def rejection(a):
    if 'error' in a:return ['solver_error']
    fail=[]
    for k,lim in [('stance_error_p95',.003),('flat_spread_p95',.005),('max_q_change',.8),('velocity_ratio_max',1.001),('upper_joint_change',1e-7),('joint_limit_violation',1e-6),('global_lift_m',.003),('max_root_z_change',.031)]:
        if a[k] is not None and a[k]>lim:fail.append(k)
    if a['new_leg_acc_max']>max(40.,1.5*a['old_leg_acc_max']):fail.append('acceleration')
    if a['min_sole']<-.0005:fail.append('penetration')
    return fail

def contact():
    from tw75_sampling import balance
    from build_contact_references import build
    original=json.loads((STATE/'source_manifest.json').read_text());lookup={r['id']:r for r in original['motions']}
    m=json.loads((STATE/'fk_manifest.json').read_text());rows=[r for r in m['motions'] if r['quality_pass'] and r['fk_quality_pass']]
    audits=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        for a in pool.map(correct,rows,chunksize=1):
            audits.append(a)
            if len(audits)%25==0:print('contact',len(audits),'/',len(rows),flush=True)
            (STATE/'contact_audit.json').write_text(json.dumps(audits,indent=2))
    byid={r['id']:r for r in rows};chosen=[]
    for a in audits:
        why=rejection(a);a['rejection_reasons']=why
        if why:continue
        r=byid[a['id']];old=lookup[r['id']]
        chosen.append(dict(r,file=str(STATE/'corrected_final/train'/(r['id']+'.npz')),original_file=r['file'],category=old['category'],split=old['split'],source_family=old['source_family'],contact_ik_applied=True,contact_ik_audit=a,sole_min_z=a['min_sole']))
    (STATE/'contact_audit.json').write_text(json.dumps(audits,indent=2))
    chosen=balance(chosen);counts={c:sum(r['category']==c for r in chosen) for c in ('walking','standing_upper')}
    # Preserve the already reviewed TW195 minimum corpus requirements.
    if counts['walking']<300 or counts['standing_upper']<1000:
        raise ValueError(f'Insufficient validated mixed corpus: {counts}')
    m['motions']=chosen;m['source_manifest']=str(SOURCE);m['contact_ik']={'variant':'corrected','source':'cpp_refresh_20261024'}
    raw=STATE/'train_without_contacts.json';raw.write_text(json.dumps(m,indent=2));build(raw,STATE/'train.json')
    summary=dict(counts=counts,frames=sum(r['frames'] for r in chosen),upstream=json.loads((SOLVER/'derived_model.json').read_text()),source_manifest_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),training_manifest_sha256=hashlib.sha256((STATE/'train.json').read_bytes()).hexdigest())
    (STATE/'data_audit.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['retarget','fk','contact']);a=p.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    globals()[a.stage]()
