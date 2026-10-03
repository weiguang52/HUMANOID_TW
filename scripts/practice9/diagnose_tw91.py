"""Frozen TW84 telemetry audit; does not alter references or policies."""
import json,itertools,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_s2_joint_commands import highpass
ROOT=Path('/root/gpufree-data/datasets/practice9')
STATE=ROOT/'tw75_contact_v1'
OUT=ROOT/'tw91_slip_v1/diagnosis'
OUT.mkdir(parents=True,exist_ok=True)
URDF=ROOT/'custom_robot/urdf/urdf0711_training_30dof.urdf'

def events(force,valid,dt):
    state=False; contact=np.zeros(len(force),bool)
    for i,f in enumerate(force):
        if not valid[i]:state=False
        elif f>=1:state=True
        elif f<=.4:state=False
        contact[i]=state
    edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
    dwell={True:[],False:[]};switches=0
    for a,b in zip(edges[::2],edges[1::2]):
        change=np.flatnonzero(np.diff(contact[a:b]))+a+1
        switches+=len(change)
        # Exclude censored first/last runs of each valid segment.
        for x,y in zip(change[:-1],change[1:]):
            dwell[bool(contact[x])].append((y-x)*dt)
    return dict(switches_per_s=switches/(valid.sum()*dt),
        stance_short_fraction=float(np.mean(np.array(dwell[True])<.04)) if dwell[True] else None,
        air_short_fraction=float(np.mean(np.array(dwell[False])<.04)) if dwell[False] else None,
        stance_median_s=float(np.median(dwell[True])) if dwell[True] else None,
        air_median_s=float(np.median(dwell[False])) if dwell[False] else None)

result={'method':'200Hz physics; first 1s and reset +/-0.5s excluded. 2-8Hz band is diagnostic, not all unwanted jitter. Hysteresis on1/off0.4N; dwell excludes censored boundaries. Foot velocities are link origin velocities, not contact patch velocities.', 'replays':[], 'reference':[]}
for variant,directory in [('baseline',ROOT/'tw75_train_v1/seed123'),('control',STATE/'control'),('contact',STATE/'contact')]:
 for f in sorted(directory.glob('*.seed*.npz')):
    a=np.load(f);step=a['control_step'];dt=float(a['physics_dt']);valid=step>=50
    for end in np.flatnonzero(a['terminal']):valid &= abs(step-end)>25
    force=a['force'][:,:,2];vel=np.linalg.norm(a['foot_velocity'][:,:,:2],axis=-1)
    bands={}
    for key in ['q','target_q','torque']:
        x=a[key]*(180/np.pi if key!='torque' else 1)
        b=highpass(x,valid,1/dt,[2,8])
        bands[key]=np.sqrt(np.nanmean(b*b,axis=0)).tolist()
    row=dict(variant=variant,clip=f.stem,joint_names=a['joint_names'].tolist(),band_rms=bands,
        force_p99_n=np.quantile(force[valid],.99,axis=0).tolist(),
        hysteresis=[events(force[:,i],valid,dt) for i in range(2)],
        slip_cost_unweighted=float(np.mean(np.sum(vel*(np.linalg.norm(a['force'],axis=-1)>1),axis=1)[valid])))
    result['replays'].append(row)
root=ET.parse(URDF).getroot()
for f in sorted((STATE/'eval').glob('*.json')):
    m=json.loads(f.read_text())
    if not isinstance(m,dict) or 'motions' not in m:continue
    motion=m['motions'][0];a=np.load(motion['file'])
    labels=np.load(m['contact_reference']['file']);names=a['body_names'].tolist();feet=[]
    for side,n in enumerate(['left_foot','right_foot']):
        c=root.find("link[@name='%s']/collision"%n)
        origin=np.fromstring(c.find('origin').get('xyz'),sep=' ')
        size=np.fromstring(c.find('geometry/box').get('size'),sep=' ')
        vertices=np.array(list(itertools.product([-.5,.5],repeat=3)))*size+origin
        j=names.index(n);q=a['body_quat_w'][:,j]
        R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        height=(np.einsum('tij,vj->tvi',R,vertices)+a['body_pos_w'][:,j,None,:])[:,:,2].min(1)
        stance=labels['contact'][:,side].astype(bool)&labels['known'][:,side].astype(bool)
        speed=np.linalg.norm(a['body_lin_vel_w'][:,j,:2],axis=-1)
        feet.append(dict(known_fraction=float(labels['known'][:,side].mean()),
            stance_frames=int(stance.sum()),stance_height_p95_m=float(np.quantile(height[stance],.95)) if stance.any() else None,
            stance_speed_mean_m_s=float(speed[stance].mean()) if stance.any() else None))
    result['reference'].append(dict(clip=f.stem,feet=feet,split=motion.get('split'),source_family=motion.get('source_family')))
(OUT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
for variant in ['baseline','control','contact']:
 rr=[r for r in result['replays'] if r['variant']==variant and r['clip'][:6] in ['000039','000139','000211']]
 b=np.sqrt(np.mean([np.square(r['band_rms']['q']) for r in rr],axis=0))
 names=rr[0]['joint_names'];order=np.argsort(-b)[:5]
 print(variant,'top_q_deg',[(names[i],round(b[i],3)) for i in order])
 print('cost',np.mean([r['slip_cost_unweighted'] for r in rr]))
for r in result['reference']:print('reference',r)
