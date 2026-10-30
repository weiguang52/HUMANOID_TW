from pathlib import Path
import numpy as np,json,datetime
root=Path(__file__).parent;reports=[]
for state in [root,root.parent/'velocity_balance_seed123_20261029']:
 r=state/'evaluation';flat=json.load(open(r/'flat_three_seed_summary.json'))
 if flat.get('complete') and not flat.get('all_complete_same_duration'):
  out={'complete':False,'pairs':24,'summary':None,'reason':'Full24 matched-success aggregate unavailable after failed or unequal-duration flat; no partial comparison substituted.'};(state/'arm_tracking_diagnostic.json').write_text(json.dumps(out,indent=2));reports.append({'state':state.name,**out});continue
 manifest=json.loads((r/'manifest.json').read_text())['motions'];rows=[]
 for seed in [42,123,2026]:
  jobs=[json.loads((r/model/('jobs'+str(seed)+'.json')).read_text()) for model in ['accepted','candidate']]
  for gold,cand in zip(*jobs):
   m=manifest[gold['motion_id']]
   if m.get('evaluation_class',m['category'])!='flat_forward':continue
   for model,j in zip(['accepted','candidate'],[gold,cand]):
    x=np.load(Path(j['output']).with_suffix('.npz'));end=int(np.flatnonzero(x['terminal'])[0]);c=np.arange(50,end);ix=np.searchsorted(x['control_step'],c,side='right')-1;e=np.rad2deg(x['q'][ix]-x['reference_q'][c]);names=x['joint_names'].tolist();arms=[i for i,n in enumerate(names) if any(s in n for s in ['shoulder','elbow','wrist'])]
    rows.append({'id':m['id'],'seed':seed,'model':model,'duration_s':end*.02,'arms_tracking_rmse_deg':float(np.sqrt(np.mean(e[:,arms]**2))),'joint_tracking_rmse_deg':np.sqrt(np.mean(e*e,axis=0)).tolist(),'joint_signed_error_start_deg':e[0].tolist(),'joint_signed_error_end_deg':e[-1].tolist(),'joint_bias_deg':e.mean(0).tolist()})
 assert len(rows)==48
 summary={}
 for model in ['accepted','candidate']:
  rr=[x for x in rows if x['model']==model];summary[model]={'arms_tracking_rmse_deg_mean':float(np.mean([x['arms_tracking_rmse_deg'] for x in rr])),'joint_tracking_rmse_deg_mean':dict(zip(names,np.mean([x['joint_tracking_rmse_deg'] for x in rr],axis=0).tolist()))}
 out={'complete':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pairs':24,'rows':rows,'summary':summary,'scope':'Actual q-reference tracking error in degrees, no mean subtraction. First1s excluded, same full matched episodes as fixed gates; diagnostic only, no changed sample or threshold.'};(state/'arm_tracking_diagnostic.json').write_text(json.dumps(out,indent=2));reports.append({'state':state.name,'summary':summary})
print(json.dumps(reports,indent=2))
