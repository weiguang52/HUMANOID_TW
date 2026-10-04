"""Explainable 5-second trajectory diagnostics, NOT pretrained HumanScore."""
import json
from pathlib import Path
import numpy as np


def windows(terminal, width):
    """Never join episodes; keep short tails with their true duration."""
    start = 0
    for end in [*map(int, np.flatnonzero(terminal) + 1), len(terminal)]:
        if end <= start:
            continue
        for lo in range(start, end, width):
            yield lo, min(lo + width, end)
        start = end


def mean_or_none(values):
    return float(np.mean(values)) if np.size(values) else None


def band_rms(values, valid, dt):
    from scipy.signal import butter, sosfiltfilt
    sos=butter(4,[2.,8.],btype='bandpass',fs=1/dt,output='sos')
    boundaries=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(int)))
    sum_sq=np.zeros(values.shape[1]);count=0
    trim=round(.5/dt)
    for lo,hi in zip(boundaries[::2],boundaries[1::2]):
        if hi-lo < round(2/dt):continue
        filtered=sosfiltfilt(sos,values[lo:hi],axis=0)[trim:-trim]
        sum_sq+=(filtered**2).sum(0);count+=len(filtered)
    return (np.sqrt(sum_sq/count)*180/np.pi).tolist() if count else None


def summarize(path):
    report=json.loads(path.read_text())
    with np.load(path.with_suffix('.npz')) as f:
        a={k:f[k] for k in f.files}
    dt=float(a['physics_dt']);cdt=float(a['control_dt'])
    terminal=a['terminal'];step=a['control_step']
    upper_error=np.linalg.norm(a['upper_actual']-a['upper_reference'],axis=-1)
    force=a['force'][:,:,2];contact=force>1.
    known=a['reference_known'][step].astype(bool)
    reference=a['reference_contact'][step].astype(bool)
    speed=np.linalg.norm(a['foot_velocity'][:,:,:2],axis=-1)
    valid=~terminal[step]
    wrist=[list(a['upper_body_names']).index(n) for n in ('left_wrist','right_wrist')]
    rows=[]
    for lo,hi in windows(terminal,round(5/cdt)):
        cv=np.arange(lo,hi);cv=cv[~terminal[cv]]
        pv=(step>=lo)&(step<hi)&valid
        if not len(cv) or not pv.any():
            continue
        mask=pv[:,None]&known
        support=pv[:,None]&contact
        pairs=pv[1:]&pv[:-1]
        switch=(np.diff(contact.astype(int),axis=0)!=0)&pairs[:,None]
        actual=a['upper_actual'][cv];ref=a['upper_reference'][cv]
        ar=np.sqrt(np.mean((actual-actual.mean(0))**2,axis=(0,2)))
        rr=np.sqrt(np.mean((ref-ref.mean(0))**2,axis=(0,2)))
        delta=(a['anchor_actual'][cv[-1],:2]-a['anchor_actual'][cv[0],:2])-(a['anchor_reference'][cv[-1],:2]-a['anchor_reference'][cv[0],:2])
        rows.append(dict(start_step=lo,end_step=hi,seconds=(hi-lo)*cdt,valid_control_frames=len(cv),
            ends_at_reset=bool(terminal[hi-1]), upper_error_m=float(upper_error[cv].mean()),
            wrist_error_m=float(upper_error[cv][:,wrist].mean()),
            upper_per_body_error_m=upper_error[cv].mean(0).tolist(),
            reference_motion_rms_m=rr.tolist(),actual_motion_rms_m=ar.tolist(),
            amplitude_ratio=[float(x/y) if y>.002 else None for x,y in zip(ar,rr)],
            anchor_horizontal_drift_error_m=float(np.linalg.norm(delta)),
            contact_known_fraction=float(known[pv].mean()),
            contact_mismatch_fraction=mean_or_none((contact!=reference)[mask]),
            contact_force_switches_per_s=float(switch.sum()/(pv.sum()*dt)),
            foot_link_contact_speed_m_s=mean_or_none(speed[support]),
            foot_link_contact_travel_m=(speed*support).sum(0).tolist()))
        # Travel is an integral of speed; dt is necessary and retains within-window duration.
        rows[-1]['foot_link_contact_travel_m']=[x*dt for x in rows[-1]['foot_link_contact_travel_m']]
    counts=report['termination_counts']
    fail=sum(v for k,v in counts.items() if k not in ('motion_end','time_out'))
    mask=valid[:,None]&known;support=valid[:,None]&contact
    clean=counts.get('motion_end',0)>0 and fail==0
    r=dict(replay=path.stem,seed=report['seed'],termination_counts=counts,clean_motion_end=clean,
        upper_body_names=a['upper_body_names'].tolist(),windows=rows,
        joint_names=a['joint_names'].tolist(),
        actual_joint_2_8hz_rms_deg=band_rms(a['q'],valid,dt),
        command_joint_2_8hz_rms_deg=band_rms(a['target_q'],valid,dt),
        anchor_height_error_m=mean_or_none(np.abs(a['anchor_actual'][~terminal,2]-a['anchor_reference'][~terminal,2])),
        upper_error_m=mean_or_none(upper_error[~terminal]),
        wrist_error_m=mean_or_none(upper_error[~terminal][:,wrist]),
        contact_known_fraction=float(known[valid].mean()),
        contact_mismatch_fraction=mean_or_none((contact!=reference)[mask]),
        foot_link_contact_speed_m_s=mean_or_none(speed[support]),
        worst_window_wrist_error_m=max((x['wrist_error_m'] for x in rows),default=None))
    return r


def main():
    root=Path('/root/gpufree-data/datasets/practice9/tw100_expression_v1/validation')
    result=dict(note='Three validation clips, three evaluation seeds, one training seed. Limited pilot, not independent-test acceptance. 2-8Hz RMS includes intentional motion, not pure jitter. 5s diagnostics, not HumanScore. Foot-link speed/travel is a slip proxy and includes foot rolling. Contact force threshold 1N; unknown labels excluded and coverage reported. Terminal control samples excluded from kinematic diagnostics, but all termination counts retained. No window crosses resets. Upper-body coordinates are chest-centered yaw frames.',variants={})
    for variant in ('expression','control'):
        rows=[summarize(p) for p in sorted((root/variant).glob('*.seed*.json')) if not p.name.endswith('.summary.json')]
        assert len(rows)==9,(variant,len(rows))
        result['variants'][variant]=dict(replays=rows,clean_replays=sum(r['clean_motion_end'] for r in rows),
            mean={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in
                  ('anchor_height_error_m','upper_error_m','wrist_error_m','contact_mismatch_fraction','foot_link_contact_speed_m_s')})
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({v:{'clean':d['clean_replays'],'mean':d['mean']} for v,d in result['variants'].items()},indent=2))

if __name__=='__main__':main()
