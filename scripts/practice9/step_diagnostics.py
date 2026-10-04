"""TW140 amplitude and phase diagnostics; do not equate completion with gait."""
import numpy as np


def diagnose(path, motion, offset):
    a=np.load(path.with_suffix('.npz'))
    ref=np.load(motion['file'])
    end=int(np.flatnonzero(a['terminal'])[0]) if a['terminal'].any() else len(a['terminal'])
    cv=np.arange(50,end)
    if not len(cv):return {'first_episode_seconds':end*.02,'insufficient_samples':True}
    sel=(a['control_step']>=50)&(a['control_step']<end)
    ix=a['reference_frame'][cv].astype(int)-offset
    bn=ref['body_names'].tolist()
    fi=[bn.index(x) for x in ('left_foot','right_foot')]
    ji=[a['joint_names'].tolist().index(x) for x in
        ('left_knee_pitch_joint','right_knee_pitch_joint')]
    def span(x):return np.percentile(x,95,axis=0)-np.percentile(x,5,axis=0)
    step=a['control_step'][sel]
    known=a['reference_known'][step].astype(bool)
    label=a['reference_contact'][step].astype(bool)
    support=a['force'][sel,:,2]>1
    speed=np.linalg.norm(a['foot_velocity'][sel,:,:2],axis=-1)
    stance=known&label&support
    swing=known&~label
    def mean(x):return float(x.mean()) if x.size else None
    return dict(first_episode_seconds=end*.02,
        knee_reference_span_deg=(span(a['reference_q'][cv][:,ji])*180/np.pi).tolist(),
        knee_actual_span_deg=(span(a['q'][sel][:,ji])*180/np.pi).tolist(),
        foot_reference_z_span_mm=(span(ref['body_pos_w'][ix][:,fi,2])*1000).tolist(),
        foot_actual_z_span_mm=(span(a['foot_position'][sel,:,2])*1000).tolist(),
        known_fraction=known.mean(0).tolist(),
        known_swing_fraction=swing.mean(0).tolist(),
        known_swing_contact_fraction=mean(support[swing]),
        known_stance_contact_speed_m_s=mean(speed[stance]),
        anchor_reference_net_distance_m=float(np.linalg.norm(
            a['anchor_reference'][cv[-1],:2]-a['anchor_reference'][cv[0],:2])),
        anchor_actual_net_distance_m=float(np.linalg.norm(
            a['anchor_actual'][cv[-1],:2]-a['anchor_actual'][cv[0],:2])),
        note='First episode excluding first 1s; P95-P5 spans. Foot-link Z is not sole clearance. Contact speed includes rolling; do not label it pure slip.')
