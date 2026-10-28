"""Persist and verify opt-in training reward-filter semantics."""
from pathlib import Path
import json,math

JOINTS=[s+'_'+j+'_joint' for s in ['left','right'] for j in
        ['hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_yaw','ankle_pitch']]

def contract(cfg):
    term=getattr(cfg.rewards,'moving_leg_highpass',None)
    if term is None or term.weight==0:return None
    params=term.params;order=params.get('filter_order',1)
    if type(order) is not int or order not in (1,4):raise ValueError('Unsupported highpass order')
    dt=float(cfg.sim.dt*cfg.decimation);cutoff=float(params.get('cutoff_hz',8.));scale=float(params.get('scale',.01))
    if not math.isfinite(dt) or dt<=0 or not math.isfinite(cutoff) or not 0<cutoff<.5/dt:
        raise ValueError('Invalid highpass timing or cutoff')
    if not math.isfinite(scale) or scale<=0 or not math.isfinite(term.weight):
        raise ValueError('Invalid highpass scale or weight')
    payload={'version':1,'filter_order':order,'dt':dt,'cutoff_hz':cutoff,'scale_rad':scale,
        'weight':float(term.weight),'joint_names':JOINTS,'reference_speed_low':.05,'reference_speed_high':.1,
        'reset':'episode_length<=1; first observation initializes constant history','scope':'reward only; no policy input or action filter changes'}
    if order==4:
        from scipy.signal import butter
        payload.update(design='Butterworth SOS',sos=butter(4,cutoff,btype='highpass',fs=1./dt,output='sos').tolist())
    else:payload.update(design='exponential low subtraction',decay=math.exp(-2*math.pi*cutoff*dt))
    return payload

def validate_resume(saved,requested):
    if saved is not None and saved!=requested:
        raise ValueError('Resume must preserve checkpoint highpass reward contract')

def prepare(cfg,log_dir,checkpoint=None):
    requested=contract(cfg)
    if checkpoint:
        path=Path(checkpoint).parent/'params/highpass_reward.json'
        if path.exists():validate_resume(json.loads(path.read_text()),requested)
    if requested is not None:
        path=Path(log_dir)/'params/highpass_reward.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(requested,indent=2))
    return requested

def finish(env,log_dir,payload):
    if payload is None:return
    import numpy as np
    base=env.unwrapped;command=base.command_manager.get_term('motion')
    updates=command._leg_error_highpass_updates;trace=command._leg_error_highpass_trace
    if updates<=0:raise ValueError('No observed highpass reward updates')
    expected='ButterworthErrorHighpass' if payload['filter_order']==4 else 'LegErrorHighpass'
    assert type(command._leg_error_highpass).__name__==expected
    if trace:
        data={k:np.stack([v[k] for v in trace]) for k in trace[0]}
        assert all(np.isfinite(v).all() for v in data.values())
        data['joint_names']=np.asarray(JOINTS)
        np.savez_compressed(Path(log_dir)/'highpass_reward_trace.npz',**data)
    result={'completed':True,'updates':updates,'captured_steps':len(trace),'filter_class':expected,
        'contract':payload,'scope':'Runtime update count and captured prefix only; no full-run finite-state guarantee'}
    (Path(log_dir)/'highpass_reward_completed.json').write_text(json.dumps(result,indent=2))
