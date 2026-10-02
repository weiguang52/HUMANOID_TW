#!/usr/bin/env python3
"""Compare scoped retarget pilots at matched source phase, without simulation.

Limb direction and foot tilt are diagnostic fidelity metrics, not contact or
physical feasibility certificates. All computations run on server files.
"""
import argparse,json,collections
from pathlib import Path
import numpy as np
import pinocchio as pin
from retarget_humanml3d import DEFAULT_TRAINING_URDF

PAIRS={'left_thigh':(1,4,'left_thigh','left_mid_leg'),
       'right_thigh':(2,5,'right_thigh','right_mid_leg'),
       'left_shin':(4,7,'left_mid_leg','left_ankle'),
       'right_shin':(5,8,'right_mid_leg','right_ankle'),
       'left_upper_arm':(16,18,'left_upper_arm','left_force_arm'),
       'right_upper_arm':(17,19,'right_upper_arm','right_force_arm'),
       'left_forearm':(18,20,'left_force_arm','left_wrist'),
       'right_forearm':(19,21,'right_force_arm','right_wrist')}

def angle(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/max(norm,1e-12),-1,1))))

def summarize(root,variants):
    model=pin.buildModelFromUrdf(str(DEFAULT_TRAINING_URDF)); data=model.createData()
    fid={f.name:i for i,f in enumerate(model.frames) if f.type==pin.BODY}
    results={}
    for variant in variants:
        motions=json.loads((root/f'pilot_{variant}/retarget_manifest.json').read_text())['motions']
        errors={k:[] for k in PAIRS}; tilt=[]; yaw=[]; clip_results=[]
        counts=collections.Counter()
        for motion in motions:
            source=np.load(motion['source_file'])[:,:,[2,0,1]]
            hip=source[:,2]-source[:,1]; headings=np.arctan2(hip[:,1],hip[:,0]); ref=headings[0]
            z=np.load(motion['file'],allow_pickle=False)
            names=z['joint_names'].tolist(); angles=z['joint_pos']
            local_errors={k:[] for k in PAIRS}; local_tilt=[]; local_yaw=[]
            for phase in np.linspace(0,1,25):
                ti=int(round(phase*(len(source)-1))); qi=int(round(phase*(len(angles)-1)))
                theta=ref-headings[ti]; c,s=np.cos(theta),np.sin(theta)
                rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                human=source[ti]@rotation.T
                q=pin.neutral(model)
                for name,value in zip(names,angles[qi]):
                    joint=model.joints[model.getJointId(str(name))];q[joint.idx_q]=value
                pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
                for key,(a,b,ra,rb) in PAIRS.items():
                    predicted=data.oMf[fid[rb]].translation-data.oMf[fid[ra]].translation
                    local_errors[key].append(angle(human[b]-human[a],predicted))
                for side,a,b in [('left',7,10),('right',8,11)]:
                    pose=data.oMf[fid[f'{side}_foot']]
                    local_tilt.append(angle(pose.rotation[:,2],np.array([0,0,1])))
                    predicted=pose.rotation[:,0].copy();predicted[2]=0
                    target=human[b]-human[a];target[2]=0
                    local_yaw.append(angle(predicted,target))
            for key in errors: errors[key].append(float(np.mean(local_errors[key])))
            tilt.extend(local_tilt);yaw.extend(local_yaw)
            counts.update(k for k,v in motion['quality']['near_limit_fraction_by_joint'].items() if v>.05)
            clip_results.append({'id':motion['id'],'quality_pass':motion['quality_pass'],
                'limb_mean_angle_error_deg':{k:float(np.mean(v)) for k,v in local_errors.items()},
                'foot_tilt_p95_deg':float(np.percentile(local_tilt,95))})
        results[variant]={'total':len(motions),'quality_pass':sum(x['quality_pass'] for x in motions),
            'clipped_clips':sum(x['clipped_fraction']>0 for x in motions),
            'near_limit_clips':dict(counts.most_common()),
            'limb_error_clip_mean_p50_p90_deg':{k:np.percentile(v,[50,90]).tolist() for k,v in errors.items()},
            'foot_tilt_p50_p95_deg':np.percentile(tilt,[50,95]).tolist(),
            'foot_heading_error_p50_p95_deg':np.percentile(yaw,[50,95]).tolist(),
            'clips':clip_results}
    return {'note':'25 matched phases per clip; no temporal smoothing of source; no root/contact simulation. Foot axes use training link frame; metric is descriptive, not acceptance.', 'variants':results}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variants',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=summarize(a.root,a.variants);a.output.write_text(json.dumps(result,indent=2)+'\n')
    for name,r in result['variants'].items(): print(name,json.dumps({k:v for k,v in r.items() if k!='clips'}))
