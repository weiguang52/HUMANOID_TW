"""Attach sidecar contact supervision without changing original motion NPZs."""
from pathlib import Path
import json, hashlib
import numpy as np
from contact_labels import from_features, resample, VERSION
ROOT=Path('/root/gpufree-data/datasets/practice9')
SRC=ROOT/'tw75_train_v1'
OUT=ROOT/'tw75_contact_v1'
HUMAN=ROOT/'humanml3d_rebuild/staging-v1/HumanML3D/new_joints'

def build(source, target):
    data=json.loads(source.read_text());labels=[];known=[];audit=[];raw4=[]
    for row in data['motions']:
        a=np.load(HUMAN/(row['id']+'.npy'),allow_pickle=False)
        feature_path=HUMAN.parent/'new_joint_vecs'/(row['id']+'.npy')
        features=np.load(feature_path,allow_pickle=False)
        contact,mask,h,raw=from_features(features,a)
        raw4.append(raw[np.floor(np.linspace(0,len(raw)-1,int(row['frames']))).astype(int)])
        contact,mask=resample(contact,mask,int(row['frames']))
        labels.append(contact);known.append(mask)
        audit.append({'id':row['id'],'category':row.get('category'),'frames':len(contact),
            'known_fraction':mask.mean(axis=0).tolist(),
            'stance_fraction':contact.mean(axis=0).tolist(),
            'source_vertical_span_m':np.ptp(h,axis=0).tolist(),
            'source_contact4_mean':raw.mean(0).tolist(),
            'source_features':str(feature_path),
            'source_features_sha256':hashlib.sha256(feature_path.read_bytes()).hexdigest()})
    target.parent.mkdir(parents=True,exist_ok=True)
    bundle=target.with_suffix('.contacts.npz')
    np.savez_compressed(bundle, contact=np.concatenate(labels), known=np.concatenate(known),
        source_contact4=np.concatenate(raw4),
        clip_ids=np.array([x['id'] for x in data['motions']]),
        lengths=np.array([x['frames'] for x in data['motions']]),
        foot_names=np.array(['left_foot','right_foot']), version=np.array(VERSION))
    data['contact_reference']={'file':str(bundle),'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'version':VERSION}
    target.write_text(json.dumps(data,indent=2))
    target.with_suffix('.contact_audit.json').write_text(json.dumps(audit,indent=2))
    print(target.name,len(audit),'known',np.concatenate(known).mean(axis=0),flush=True)

if __name__=='__main__':
    for split in ['train','val','test']:
        build(SRC/'splits'/f'{split}.json',OUT/'splits'/f'{split}.json')
    for source in sorted((SRC/'eval').glob('*.json')):
        build(source,OUT/'eval'/source.name)
