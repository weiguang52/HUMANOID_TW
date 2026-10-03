"""Contact diagnostics from existing 200 Hz telemetry, excluding reset neighborhoods."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/gpufree-data/datasets/practice9')
OUT=ROOT/'tw75_contact_v1';rows=[]
for seed in [42,123]:
    for motion in ['000039','000139','000211','000016','000124','000346']:
        for evseed in [42,123,2026]:
            path=ROOT/f'tw75_train_v1/seed{seed}/{motion}.seed{evseed}.npz'
            a=np.load(path);dt=float(a['physics_dt']);step=a['control_step']
            f=a['force'][:,:,2];speed=np.linalg.norm(a['foot_velocity'][:,:,:2],axis=-1)
            valid=step>=50
            for t in np.flatnonzero(a['terminal']):valid &= np.abs(step-t)>25
            row={'model_seed':seed,'motion':motion,'eval_seed':evseed,'seconds':float(valid.sum()*dt),'feet':[]}
            for side in range(2):
                raw=f[:,side]>1.;stable=np.zeros(len(f),dtype=bool);state=False
                for t,v in enumerate(f[:,side]):
                    if v>=1.:state=True
                    elif v<.4:state=False
                    stable[t]=state
                pair=valid[1:]&valid[:-1]
                mask=raw&valid
                row['feet'].append({'raw_switches_per_s':float(np.sum((raw[1:]!=raw[:-1])&pair)/(valid.sum()*dt)),
                    'hysteresis_switches_per_s':float(np.sum((stable[1:]!=stable[:-1])&pair)/(valid.sum()*dt)),
                    'contact_fraction':float(raw[valid].mean()),
                    'contact_slip_mean_m_s':float(speed[mask,side].mean()),
                    'contact_slip_p95_m_s':float(np.percentile(speed[mask,side],95))})
            rows.append(row)
            if seed==123 and evseed==42 and motion in ['000039','000211']:
                b=np.load(OUT/f'eval/{motion}.contacts.npz')
                t=np.arange(len(f))*dt;sel=(t>=3)&(t<=9)
                fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
                for side,color in [(0,'tab:blue'),(1,'tab:orange')]:
                    axes[0].plot(t[sel],f[sel,side],color=color,label=['left','right'][side],lw=.8)
                    axes[1].plot(t[sel],speed[sel,side],color=color,lw=.8)
                    ref_t=np.arange(len(b['contact']))*.02
                    target=np.where(b['known'][:,side],b['contact'][:,side],np.nan)
                    axes[2].step(ref_t,target+.05*side,color=color,where='post')
                axes[0].axhline(1,color='k',ls='--',lw=.7);axes[0].legend()
                axes[0].set_ylabel('Vertical force [N]');axes[1].set_ylabel('Foot XY speed [m/s]')
                axes[2].set_ylabel('Human phase: stance=1');axes[2].set_xlabel('Simulation time [s]')
                axes[2].set_xlim(3,9);fig.suptitle(motion+' / model123 / eval42; gaps = uncertain labels')
                fig.tight_layout();fig.savefig(OUT/(motion+'.contact.png'),dpi=160);plt.close(fig)
(OUT/'baseline_contacts.json').write_text(json.dumps(rows,indent=2))
print('Saved',len(rows),'diagnostics and 2 plots')
