"""Per-joint command jitter in degrees, excluding reset/filter edge effects."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt,welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/gpufree-data/datasets/practice9/s2_trials')
OUT=Path('/root/gpufree-data/datasets/practice9/s2_joint_analysis')
IDS=['000801','006680','010407','010782']
OUT.mkdir(parents=True,exist_ok=True)

def highpass(x,valid,rate,band=None):
    result=np.full_like(x,np.nan,dtype=float)
    edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
    sos=butter(4,8 if band is None else band,fs=rate,btype='highpass' if band is None else 'bandpass',output='sos')
    for a,b in zip(edges[::2],edges[1::2]):
        trim=int(.5*rate)
        if b-a<2*rate:continue
        filtered=sosfiltfilt(sos,x[a:b],axis=0)
        result[a+trim:b-trim]=filtered[trim:-trim]
    return result

def read(path):
    d=np.load(path);step=d['control_step'];ends=np.r_[np.flatnonzero(np.diff(step)),len(step)-1]
    valid=~d['terminal'].copy();valid[0]=False
    for shift in (-1,1): valid &= ~np.roll(d['terminal'],shift)
    rate=1/float(d['control_dt']);physics_rate=1/float(d['physics_dt'])
    target=np.rad2deg(d['target_q'][ends]);actual=np.rad2deg(d['q'][ends]);reference=np.rad2deg(d['reference_q'])
    hf=highpass(target,valid,rate)
    actual_hf=highpass(np.rad2deg(d['q']),valid[step],physics_rate)
    ref_hf=highpass(reference,valid,rate)
    delta=np.diff(target,axis=0);delta[~(valid[1:]&valid[:-1])]=np.nan
    band=highpass(target,valid,rate,[2,8]);actual_band=highpass(np.rad2deg(d['q']),valid[step],physics_rate,[2,8]);ref_band=highpass(reference,valid,rate,[2,8])
    return dict(band=band,actual_band=actual_band,ref_band=ref_band,names=d['joint_names'].tolist(),target=target,actual=actual,reference=reference,hf=hf,actual_hf=actual_hf,ref_hf=ref_hf,delta=delta,valid=valid,rate=rate)

if __name__ == '__main__':
    reports={};all_data={}
    for variant in ['baseline','smooth']:
        clips=[read(ROOT/variant/(i+'.telemetry.npz')) for i in IDS];all_data[variant]=clips
        names=clips[0]['names'];hf=np.concatenate([d['hf'] for d in clips]);actual=np.concatenate([d['actual_hf'] for d in clips]);ref=np.concatenate([d['ref_hf'] for d in clips]);delta=np.concatenate([d['delta'] for d in clips])
        band=np.concatenate([d["band"] for d in clips]);actual_band=np.concatenate([d["actual_band"] for d in clips]);ref_band=np.concatenate([d["ref_band"] for d in clips])
        rows=[]
        for i,name in enumerate(names):
            rows.append(dict(joint=name,command_2_8_rms_deg=float(np.sqrt(np.nanmean(band[:,i]**2))),actual_2_8_rms_deg=float(np.sqrt(np.nanmean(actual_band[:,i]**2))),reference_2_8_rms_deg=float(np.sqrt(np.nanmean(ref_band[:,i]**2))),command_hf_rms_deg=float(np.sqrt(np.nanmean(hf[:,i]**2))),command_hf_abs_p95_deg=float(np.nanquantile(abs(hf[:,i]),.95)),command_hf_abs_max_deg=float(np.nanmax(abs(hf[:,i]))),actual_hf_rms_deg=float(np.sqrt(np.nanmean(actual[:,i]**2))),reference_hf_rms_deg=float(np.sqrt(np.nanmean(ref[:,i]**2))),step_abs_p95_deg=float(np.nanquantile(abs(delta[:,i]),.95)),step_abs_max_deg=float(np.nanmax(abs(delta[:,i])))))
        reports[variant]=sorted(rows,key=lambda r:r['command_2_8_rms_deg'],reverse=True)
    summary=dict(method='Four walking clips, seed42, observation noise off. Actual actuator target angles in degrees; not raw normalized policy actions. 4th-order zero-phase 2-8Hz bandpass and 8Hz highpass, per valid segment, exclude terminal +/-1 control step and 0.5s filter edges. Actual angle sampled at200Hz, command/reference at50Hz. RMS pooled by valid sample count. Highpass choice is a diagnostic, not a perceptual threshold.',results=reports)
    (OUT/'joint_jitter_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    order=[r['joint'] for r in reports['smooth']];base={r['joint']:r for r in reports['baseline']};smooth={r['joint']:r for r in reports['smooth']}
    fig,axes=plt.subplots(1,2,figsize=(17,13),layout='constrained');y=np.arange(len(order))
    for ax,key,title in [(axes[0],'command_2_8_rms_deg','Command oscillation: 2-8Hz RMS'),(axes[1],'actual_2_8_rms_deg','Actual joint oscillation: 2-8Hz RMS')]:
     ax.barh(y-.18,[base[n][key] for n in order],height=.35,label='Baseline');ax.barh(y+.18,[smooth[n][key] for n in order],height=.35,label='Smooth');ax.set_yticks(y,order);ax.invert_yaxis();ax.set_xlabel('degrees RMS');ax.set_title(title);ax.legend();ax.grid(axis='x',alpha=.2)
    fig.suptitle('S2 walking: all 30 joints / 4 clips / seed 42');fig.savefig(OUT/'all_joint_2_8hz.png',dpi=150);plt.close(fig)
    for ci,clip in enumerate(IDS):
     d=all_data['smooth'][ci];t=np.arange(len(d['target']))/d['rate']
     for page in range(3):
      fig,axes=plt.subplots(5,2,figsize=(16,14),sharex=True,layout='constrained')
      for ax,j in zip(axes.flat,range(page*10,(page+1)*10)):
       for key,label,color in [('reference','Reference','black'),('target','Command','tab:orange'),('actual','Actual','tab:blue')]:
        values=d[key][:,j].copy();values[~d['valid']]=np.nan;ax.plot(t,values,lw=.65,label=label,color=color,alpha=.8)
       ax.set_title(d['names'][j],fontsize=10);ax.set_ylabel('deg');ax.grid(alpha=.2)
      axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('seconds');axes[-1,1].set_xlabel('seconds')
      fig.suptitle(f'Smooth policy / walk {clip} / joints {page*10+1}-{page*10+10}');fig.savefig(OUT/f'{clip}_commands_{page+1}.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(15,12),layout='constrained')
    for ax,n in zip(axes,order[:5]):
     for v in ['baseline','smooth']:
      d=all_data[v][0];j=d['names'].index(n);t=np.arange(len(d['hf']))/d['rate'];mask=(t>=2)&(t<=4);ax.plot(t[mask],d['band'][mask,j],label=v,lw=.9)
     ax.set_title(n);ax.set_ylabel('2-8Hz command deg');ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('seconds; clip 000801');fig.savefig(OUT/'top5_2_8hz_zoom.png',dpi=150);plt.close(fig)
    print(json.dumps(reports['smooth'][:8],indent=2))
