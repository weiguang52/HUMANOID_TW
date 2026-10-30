from pathlib import Path
import numpy as np,json,datetime
from scipy.signal import butter,sosfilt,sosfilt_zi
b=Path(__file__).parent;sos=butter(4,8,btype='highpass',fs=50,output='sos');reports=[]
for state in [b,b.parent/'anchor_strength_seed123_20261028']:
 rows=[];manifest=json.load(open(state/'evaluation/manifest.json'))['motions']
 for seed in [42,123,2026]:
  for model in ['accepted','candidate']:
   for j in json.load(open(state/'evaluation'/model/f'jobs{seed}.json')):
    m=manifest[j['motion_id']]
    if m.get('evaluation_class',m['category'])!='flat_forward':continue
    with np.load(Path(j['output']).with_suffix('.npz')) as x:
     end=int(np.flatnonzero(x['terminal'])[0]);cc=np.arange(end);ix=np.searchsorted(x['control_step'],cc,side='right')-1;e=np.rad2deg(x['q'][ix]-x['reference_q'][cc]);v=e[50:];freq=np.fft.rfftfreq(len(v),.02);f=np.fft.rfft(v-v.mean(0),axis=0);f[freq<8]=0;rect=np.fft.irfft(f,n=len(v),axis=0);w=np.hanning(len(v));f=np.fft.rfft((v-v.mean(0))*w[:,None],axis=0);f[freq<8]=0;hann=np.fft.irfft(f,n=len(v),axis=0)/np.sqrt(np.mean(w*w));causal=sosfilt(sos,e,axis=0,zi=sosfilt_zi(sos)[:,:,None]*e[0][None,None,:])[0][50:];names=x['joint_names'].tolist()
     groups={'legs12':[i for i,n in enumerate(names) if any(k in n for k in ['hip','knee','ankle'])],'arms10':[i for i,n in enumerate(names) if any(k in n for k in ['shoulder','elbow','wrist'])],'waist3':[i for i,n in enumerate(names) if 'waist' in n],'all30':list(range(30))}
     for group,ids in groups.items():
      rows.append({'model':model,'seed':seed,'id':m['id'],'group':group,'rect_rms_deg':float(np.sqrt(np.mean(rect[:,ids]**2))),'hann_rms_deg':float(np.sqrt(np.mean(hann[:,ids]**2))),'causal_rms_deg':float(np.sqrt(np.mean(causal[:,ids]**2))),'endpoint_jump_deg':float(np.sqrt(np.mean((v[-1,ids]-v[0,ids])**2)))})
 assert len(rows)==192
 summary={g:{model:{k:float(np.mean([r[k] for r in rows if r['group']==g and r['model']==model])) for k in ['rect_rms_deg','hann_rms_deg','causal_rms_deg','endpoint_jump_deg']} for model in ['accepted','candidate']} for g in groups}
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'pairs':24,'summary':summary,'rows':rows,'scope':'Supplemental all24 matched trajectories, exclude first1s. Original rectangular FFT stays canonical; causal/Hann diagnostics do not replace thresholds or prove endpoint causality.'};(state/'fullbody_spectrum_diagnostic.json').write_text(json.dumps(out,indent=2));reports.append({'state':state.name,'summary':summary})
print(json.dumps(reports,indent=2))
