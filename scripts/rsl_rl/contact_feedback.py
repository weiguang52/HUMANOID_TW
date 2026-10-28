"""Opt-in actual-contact observation features; default policy inputs unchanged."""
import math
import torch


def contact_statistics(forces, points, counts, starts, com, linear, angular, capacity=64):
    """Per-environment load, load-weighted point velocity, and unweighted XY RMS.

    The one-foot/one-ground-filter PhysX buffer contract matches point_slip.py.
    Velocity includes angular motion about body COM. No reference labels used.
    """
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError('Positive integer capacity required')
    n=com.shape[0]
    if any(x.shape != (n,3) for x in [com,linear,angular]) or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('Expected Nx3 body vectors and Px3 points')
    counts,starts=counts.reshape(-1),starts.reshape(-1)
    if counts.numel()!=n or starts.numel()!=n or forces.numel()!=len(points):
        raise ValueError('Expected one foot and ground filter per environment')
    if counts.dtype.is_floating_point or starts.dtype.is_floating_point:
        raise ValueError('Counts and starts must be integers')
    if (counts<0).any() or (counts>capacity).any() or (starts<0).any():
        raise ValueError('Invalid or truncated contact buffer')
    if ((counts>0)&(starts+counts>len(points))).any():
        raise ValueError('Contact ranges outside point buffer')
    if not all(torch.isfinite(x).all() for x in [com,linear,angular]):
        raise ValueError('Nonfinite body state')
    if len(points)==0:
        return {'load':com.new_zeros(n),'velocity':com.new_zeros(n,3),'xy_rms':com.new_zeros(n)}
    offsets=torch.arange(capacity,device=com.device)[None,:]
    valid=offsets<counts[:,None]
    indices=(starts[:,None].long()+offsets).clamp(0,len(points)-1)
    f=torch.where(valid,forces.reshape(-1)[indices],0.)
    if not torch.isfinite(f).all():raise ValueError('Nonfinite active contact force')
    valid=valid&(f>1e-6)
    f=torch.where(valid,f,0.)
    p=torch.where(valid[...,None],points[indices],com[:,None,:])
    if not torch.isfinite(p).all():raise ValueError('Nonfinite active contact point')
    v=linear[:,None,:]+torch.cross(angular[:,None,:].expand_as(p),p-com[:,None,:],dim=-1)
    load=f.sum(-1)
    mean=(v*f[...,None]).sum(1)/load.clamp_min(1e-12)[:,None]
    square=torch.where(valid,v[...,:2].square().sum(-1),0.)
    rms=(square.sum(-1)/valid.sum(-1).clamp_min(1)).sqrt()
    return {'load':load,'velocity':torch.where((load>0)[:,None],mean,0.),'xy_rms':rms}


def contact_features(load, velocity, xy_rms, base_quat_wxyz, load_scale=100., speed_scale=.1):
    """Eight bounded features: [load,vx_heading,vy_heading,XY_RMS] per foot.

    Loads are N, speeds m/s; no-contact speeds are zero below 5 N. Fixed
    normalizations define the version1 optional policy input contract.
    """
    if not all(math.isfinite(v) and v>0 for v in [load_scale,speed_scale]):
        raise ValueError('Positive finite scales required')
    if load.ndim!=2 or load.shape[1]!=2:raise ValueError('Expected Nx2 loads')
    n=len(load)
    if velocity.shape!=(n,2,3) or xy_rms.shape!=(n,2) or base_quat_wxyz.shape!=(n,4):
        raise ValueError('Invalid contact feature dimensions')
    if not all(torch.isfinite(x).all() for x in [load,velocity,xy_rms,base_quat_wxyz]):
        raise ValueError('Nonfinite feature input')
    if (load<0).any() or (xy_rms<0).any():raise ValueError('Negative load or RMS')
    norm=base_quat_wxyz.norm(dim=-1,keepdim=True)
    if (norm<1e-8).any():raise ValueError('Invalid base quaternion')
    w,x,y,z=(base_quat_wxyz/norm).unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));c=yaw.cos()[:,None];s=yaw.sin()[:,None]
    vx=c*velocity[:,:,0]+s*velocity[:,:,1];vy=-s*velocity[:,:,0]+c*velocity[:,:,1]
    loaded=load>5.
    out=torch.stack([(load/load_scale).clamp(0,3),torch.where(loaded,vx/speed_scale,0.).clamp(-5,5),torch.where(loaded,vy/speed_scale,0.).clamp(-5,5),torch.where(loaded,xy_rms/speed_scale,0.).clamp(0,5)],dim=-1)
    return out.reshape(n,8)


def ground_contact_features(env, sensor_prefix="slip_"):
    """Read current ground sensors; reset observations discard stale contacts.

    Optional observation wrapper and shadow diagnostics call this explicitly.
    """
    if sensor_prefix not in ['slip_','diagnostic_']:raise ValueError('Unknown contact sensor prefix')
    robot=env.scene['robot'];data=robot.data;stats=[]
    for name in ['left_foot','right_foot']:
        body=robot.body_names.index(name);sensor=env.scene[sensor_prefix+name]
        force,point,_,_,count,start=sensor.contact_physx_view.get_contact_data(dt=env.cfg.sim.dt)
        stats.append(contact_statistics(force,point,count,start,data.body_com_pos_w[:,body],data.body_lin_vel_w[:,body],data.body_ang_vel_w[:,body]))
    features=contact_features(torch.stack([v['load'] for v in stats],dim=1),torch.stack([v['velocity'] for v in stats],dim=1),torch.stack([v['xy_rms'] for v in stats],dim=1),data.root_quat_w)
    return torch.where((env.episode_length_buf>0)[:,None],features,0.)
