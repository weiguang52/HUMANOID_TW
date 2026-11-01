import pathlib,subprocess,sys,json,numpy as np,hashlib,datetime
s=pathlib.Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/balance_authority_v1');o=s/'cold_repeats';assert not o.exists();o.mkdir();rows=[];(o/'status').write_text('collecting\n')
try:
 for case in range(32):
  for variant in [0,1]:
   dest=o/('case%02d_variant%d'%(case,variant));log=o/('case%02d_variant%d.log'%(case,variant))
   with log.open('w') as stream:subprocess.run([sys.executable,'scripts/practice9/probe_supermimic_balance_authority_serial.py','--plan',str(s/'coincident_plan.json'),'--case-index',str(case),'--variant-index',str(variant),'--output',str(dest)],stdout=stream,stderr=subprocess.STDOUT,check=True)
  with np.load(o/('case%02d_variant0'%case)/'trace.npz') as x,np.load(o/('case%02d_variant1'%case)/'trace.npz') as y:
   assert all(x[k].shape==y[k].shape for k in x.files);error=max(float(abs(x[k].astype(float)-y[k].astype(float)).max()) for k in x.files);row=dict(case=case,max_error=error,passed=error<=1e-5);rows.append(row);(o/'partial.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n');print(row,flush=True)
   if not row['passed']:raise RuntimeError('Unchanged1e-5 repeatability gate failed; preserve outputs')
 (o/'audit.json').write_text(json.dumps(dict(passed=True,scope='All32 cold serial zero-repeat pairs only; no perturbed response or feedback controller accepted.',rows=rows),indent=2)+'\n');(o/'status').write_text('all32_cold_repeats_passed_pending_perturbations\n')
except BaseException:
 (o/'status').write_text('failed_preserve_all_outputs\n');raise
