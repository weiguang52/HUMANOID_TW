"""Diagnose a failed fit line search; optional stricter foot-equality smoke, no training export."""
import argparse, hashlib, json, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for key in ['plan','asset','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();plan=json.loads(a.plan.read_text())
assert plan['source_split']=='fit'
assert plan['numerics']['iterations']==(2 if plan.get('strict_linear_foot_equalities') else 1)
assert plan['replay_source_id'] in plan['initial_correction_sha256']
source=Path(__file__).with_name('solve_supermimic_grounded_dynamic_cop.py')
s=source.read_text();assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['solver_source_sha256']
def replace(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new)
replace(" f=Path(item['file']);", " if item['id']!=plan['replay_source_id']:continue\n f=Path(item['file']);")
replace('  accepted=False','  accepted=False;trials=[];entry[\'line_search_trials\']=trials\n  sp.save_npz(dest/\'replay_P.npz\',P);sp.save_npz(dest/\'replay_A.npz\',A);np.savez_compressed(dest/\'replay_vectors.npz\',gradient=gradient,lower=l,upper=u,step=step,x=x,column_scale=cs,row_scale=rs,objective_scale=objective_scale)')
replace('   if bound_error>1e-7:continue', '''   components={'lower':lo-candidate,'upper':candidate-hi,'speed_negative':-.06-original_delta-V@candidate,'speed_positive':V@candidate-(.06-original_delta),'trust':abs(candidate-x)-trust}
   binding=max(components,key=lambda k:float(components[k].max()));idx=int(np.argmax(components[binding]));width=14 if binding.startswith('speed') else 17;component=idx%width
   joint=names[lower[component]] if width==14 else (['root_x','root_y','root_z']+[names[j] for j in lower])[component]
   trial=dict(alpha=alpha,bound_error=bound_error,binding= binding,binding_frame=idx//width,binding_coordinate=joint,binding_violation=float(components[binding][idx]));trials.append(trial)
   if bound_error>1e-7:trial['rejected']='bounds_speed_or_trust';continue''')
replace('   if ch.min()<-.0005:continue',"   trial['collision_min_m']=float(ch.min())\n   if ch.min()<-.0005:trial['rejected']='collision';continue")
replace('   if pe>1e-5 or oe>1e-4:continue',"   trial.update(foot_position_max_m=pe,foot_orientation_max_rad=oe)\n   if pe>1e-5 or oe>1e-4:trial['rejected']='foot_vectors';continue")
replace('   if np.any(nforce[mask,2]<=0):continue',"   trial['known_fz_min_n']=float(nforce[mask,2].min())\n   if np.any(nforce[mask,2]<=0):trial['rejected']='required_normal_force';continue")
replace('   if new<old-1e-9:',"   trial.update(merit_after=new,merit_change=new-old);trial['rejected']='merit' if new>=old-1e-9 else None\n   if new<old-1e-9:")
if plan.get('strict_linear_foot_equalities'):
 # Stricter linear equalities inside original foot tolerances; other bounds/objective unchanged.
 replace('-ftol-fr[active]', '-fr[active]')
 replace('ftol-fr[active]', '-fr[active]')
# Default replay adds observability only; explicit equality smoke is a stricter QP. Plan hashes
# bind original solver and exact last accepted correction; no output is promoted.
sys.argv=[str(source),'--plan',str(a.plan),'--asset',str(a.asset),'--output',str(a.output)]
exec(compile(s,str(source)+'[instrumented_replay]','exec'))
