from pathlib import Path
import numpy as np,json,datetime
b=Path(__file__).parent
for state in [b,b.parent/'anchor_strength_seed123_20261028']:
 rows=[];manifest=json.load(open(state/'evaluation/manifest.json'))['motions']
 for seed in [42,123,2026]:
  for model in ['accepted','candidate']:
   for j in json.load(open(state/'evaluation'/model/f'jobs{seed}.json')):
    m=manifest[j['motion_id']]
    if m.get('evaluation_class',m['category'])!='flat_forward':continue
    with np.load(Path(j['output']).with_suffix('.npz')) as x:
     end=int(np.flatnonzero(x['terminal'])[0]);cc=np.arange(50,end);ix=np.searchsorted(x['control_step'],cc,side='right')-1;q=x['pelvis_actual_quat'][cc].astype(float);r=x['pelvis_reference_quat'][cc].astype(float);assert np.max(abs(np.linalg.norm(q,axis=1)-1))<1e-4 and np.max(abs(np.linalg.norm(r,axis=1)-1))<1e-4
     yaw=lambda a:np.arctan2(2*(a[:,0]*a[:,3]+a[:,1]*a[:,2]),1-2*(a[:,2]**2+a[:,3]**2))
     dy=np.rad2deg(np.arctan2(np.sin(yaw(q)-yaw(r)),np.cos(yaw(q)-yaw(r))));angle=np.rad2deg(2*np.arccos(np.clip(abs((q*r).sum(1)),-1,1)));names=x['joint_names'].tolist();waist=[i for i,n in enumerate(names) if 'waist' in n];e=np.rad2deg(x['q'][ix]-x['reference_q'][cc]);upper=np.linalg.norm(x['upper_actual'][cc]-x['upper_reference'][cc],axis=2)*1000;wrists=[i for i,n in enumerate(x['upper_body_names'].tolist()) if 'wrist' in n]
     rows.append({'seed':seed,'model':model,'id':m['id'],'pelvis_orientation_rmse_deg':float(np.sqrt(np.mean(angle**2))),'pelvis_yaw_rmse_deg':float(np.sqrt(np.mean(dy**2))),'pelvis_yaw_signed_mean_deg':float(dy.mean()),'waist_joint_rmse_deg':float(np.sqrt(np.mean(e[:,waist]**2))),'upper6_yaw_frame_mean_error_mm':float(upper.mean()),'wrists_yaw_frame_mean_error_mm':float(upper[:,wrists].mean())})
 assert len(rows)==48
 keys=[k for k in rows[0] if k not in ['seed','model','id']];summary={model:{k:float(np.mean([r[k] for r in rows if r['model']==model])) for k in keys} for model in ['accepted','candidate']};out={'complete':True,'pairs':24,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summary':summary,'rows':rows,'scope':'All24 full first episodes exclude1s; pelvis world orientation wxyz. Upper positions are each own anchor translation/yaw removed per mimic_telemetry source; not world hand error. Descriptive, not causal or new gate.'};(state/'posture_fidelity_diagnostic.json').write_text(json.dumps(out,indent=2));print(state.name,summary)
