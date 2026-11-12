"""Issue16 scientific figures from audited observed data; never substitute missing endpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from issue16_common import ROOT,sha
MAIN=['stand','forward_03','forward_05','forward_08']
NODES=['robust05','requested4500','control6000']
ABS='full20s_survivor_heading_mean_abs_degrees'
SIGNED='full20s_survivor_heading_signed_mean_degrees'
SOURCE_SHA=sha(__file__)
def load(n):return json.loads((ROOT/n).read_text())
def main():
 assert (ROOT/'initial_postprocessing_complete').exists()
 out=ROOT/'figures';out.mkdir(exist_ok=True);outputs=[]
 def save(fig,name):
  fig.tight_layout()
  for suffix in ['.png','.svg']:
   p=out/(name+suffix);fig.savefig(p,dpi=150,bbox_inches='tight');outputs.append(str(p.relative_to(ROOT)))
  plt.close(fig)
 gate=load('lineage_heading_gate.json');comparisons=load('mirror_comparisons.json');by={(r['node'],r['command']):r for r in comparisons}
 order=list(gate['all_four_heading_gate_by_node']);rows={(r['node'],r['command']):r for r in gate['rows']}
 fig,ax=plt.subplots(figsize=(13,10));a=np.full((len(order),4),np.nan)
 for i,n in enumerate(order):
  for j,c in enumerate(MAIN):
   r=rows[n,c];v=r['mean_absolute20s_degrees']
   if v is not None:a[i,j]=0 if v<=30 else 1
 image=ax.imshow(a,cmap=ListedColormap(['#b8e0bf','#edb1a8']),vmin=0,vmax=1,aspect='auto')
 image.cmap.set_bad('#dddddd')
 for i,n in enumerate(order):
  for j,c in enumerate(MAIN):
   r=rows[n,c];v=r['mean_absolute20s_degrees'];u=r['training_bootstrap95_upper']
   text='NA' if v is None else f"{v:.1f}deg\nU95="+('NA (n=1)' if u is None else f"{u:.1f}")
   ax.text(j,i,text,ha='center',va='center',fontsize=9)
 ax.set_xticks(range(4),MAIN);ax.set_yticks(range(len(order)),[f"{n} (train n={rows[n,'stand']['training_seeds']})" for n in order])
 ax.set_title('Frozen30deg gate on original issue15 data: green=mean pass, red=mean fail, gray=missing\nNo new lineage evaluation. Upper95 is reported separately; survivors/falls remain explicit in the report.')
 save(fig,'lineage_heading_gate')
 for metric,name,title in [(SIGNED,'mirror_signed_heading','Signed reset-aware20s heading drift'),(ABS,'mirror_absolute_heading','Mean absolute reset-aware20s heading drift')]:
  fig,axes=plt.subplots(3,4,figsize=(19,12))
  for i,node in enumerate(NODES):
   for j,command in enumerate(MAIN):
    ax=axes[i,j];r=by[node,command]
    for x,key,color in [(0,'original','#377eb8'),(1,'mirrored','#e6550d')]:
     value=r[key]['metrics'][metric]
     if value is None:ax.text(x,.5,'NO COMPLETE\nENDPOINT',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
     else:
      mean=value['mean'];ci=value['bootstrap95']
      if ci:ax.errorbar(x,mean,yerr=np.array([[mean-ci[0]],[ci[1]-mean]]),fmt='o',color=color,capsize=4)
      else:ax.plot(x,mean,'s',color=color)
    for k,seed in enumerate(r['per_seed']):
     values=[seed['original_signed' if metric==SIGNED else 'original_abs'],seed['mirrored_signed' if metric==SIGNED else 'mirrored_abs']]
     if all(v is not None for v in values):ax.plot([-.04+k*.02,.96+k*.02],values,'o-',color=f'C{k}',alpha=.45,lw=.7,ms=3)
    ax.axhline(0,color='black',lw=.6)
    if metric==ABS:ax.axhline(30,color='green',ls='--',lw=.8)
    ax.set_xticks([0,1],['original','mirror']);ax.set_xlim(-.3,1.3);ax.grid(alpha=.2)
    ax.set_title(f"{node} | {command} | train n={len(r['training_seed_ids'])}",fontsize=10)
    ax.set_ylabel('degrees')
  fig.suptitle(title+'\nDots/lines are matched existing training seeds; intervals are unadjusted training-seed bootstrap95. No CI for n=1; failed endpoints are not invented.',fontsize=12)
  save(fig,name)
 fig,axes=plt.subplots(2,4,figsize=(19,10))
 for j,command in enumerate(MAIN):
  for i,node in enumerate(NODES):
   r=by[node,command];s=r.get('paired_mirrored_plus_original_signed_degrees')
   ax=axes[0,j]
   if s is None:ax.text(i,.5,'NA',transform=ax.get_xaxis_transform(),ha='center')
   else:
    ci=s['bootstrap95'];v=s['mean'];ax.errorbar(i,v,yerr=np.array([[v-ci[0]],[ci[1]-v]]) if ci else None,fmt='o' if ci else 's',capsize=4,color=f'C{i}')
   ax=axes[1,j];ratio=r.get('mean_absolute_mirrored_over_original');ci=r.get('ratio_training_bootstrap95')
   if ratio is not None and ratio>0:ax.errorbar(i,ratio,yerr=np.array([[ratio-ci[0]],[ci[1]-ratio]]) if ci else None,fmt='o' if ci else 's',capsize=4,color=f'C{i}')
   else:ax.text(i,.5,'NA',transform=ax.get_xaxis_transform(),ha='center')
  axes[0,j].axhline(0,color='black',ls='--');axes[0,j].set_ylabel('mirror+original signed drift (deg)')
  axes[1,j].axhspan(.8,1.2,color='green',alpha=.15);axes[1,j].axhline(1,color='black',ls='--');axes[1,j].set_yscale('log');axes[1,j].set_ylabel('mean abs mirror/original (log scale)')
  for i in [0,1]:
   axes[i,j].set_xticks(range(3),['robust05 n1','req4500 n5','control n5'],rotation=20);axes[i,j].set_title(command);axes[i,j].grid(alpha=.2)
 fig.suptitle('Paired training-seed diagnostics: zero signed sum is a reference, not an equivalence proof\nThe0.8..1.2 magnitude band is descriptive; missing/censored endpoints stay NA; intervals unadjusted.')
 save(fig,'paired_heading_sum_and_ratio')
 physical=load('usd_symmetry_audit.json');body=physical['body_comparisons'];labels=[r['body'].replace('left_','L_')+(' [central]' if r['central_self'] else '') for r in body]
 fig,axes=plt.subplots(2,3,figsize=(20,14))
 metrics=[('mass_relative_difference',100,'Mass difference (%)'),('COM_reflection_distance_m',1000,'COM reflection residual (mm)'),('inertia_frobenius_relative_difference',100,'Inertia Frobenius relative residual (%)'),('inertia_frobenius_absolute_difference_kg_m2',1e6,'Inertia absolute residual (x1e-6 kg m2)'),('collision_vertex_cloud_hausdorff_m',1000,'Authored collision vertex-cloud distance (mm)')]
 for ax,(key,factor,title) in zip(axes.flat,metrics):
  values=[r.get(key,0)*factor for r in body];ax.barh(range(len(body)),values,color='#756bb1');ax.set_yticks(range(len(body)),labels,fontsize=8);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2)
 axes[1,2].axis('off');axes[1,2].text(0,1,'Exact USD source audit\n\n31 rigid bodies,30 revolute joints\n31 authored meshes +2 analytic sole boxes\n\nReflected hard limits: all match\nDefault angle mapping: exact\nConfigured PD/armature/caps: all pairs equal\n\nEffective PhysX readback is a separate audit.\nVertex-cloud distances depend on tessellation;\nsource convex hulls are not extracted cooked hulls.\n\nSmall structural differences alone\ndo not prove a causal heading contribution.',va='top',fontsize=12)
 fig.suptitle('Authored robot reflection differences in a common frame\nCentral links are compared to their own reflection; relative inertia differences must be read with absolute values.')
 save(fig,'usd_reflection_differences')
 default_job=load('no_policy/default_eval1.json');folder=Path(default_job['issue16_tracking']['output'])
 with np.load(folder/'stand_yaw.npz') as z:
  failure=z['first_failure_step'];h=z['heading_delta_sum']*180/np.pi;duration=z['heading_count'][1]*.02;initial=z['heading_start']*180/np.pi
 survived=failure<0;fig,axes=plt.subplots(2,2,figsize=(14,10))
 if survived.any():axes[0,0].hist(h[0,survived],bins=30,color='#3182bd');axes[0,0].axvline(float(h[0,survived].mean()),color='black',ls='--')
 else:axes[0,0].text(.5,.5,'NO20s SURVIVORS\nNo full-horizon endpoint estimate',ha='center',va='center',transform=axes[0,0].transAxes)
 axes[0,0].set_title(f"Complete20s signed heading | {survived.sum()}/1024 survivors");axes[0,0].set_xlabel('degrees')
 axes[0,1].hist(h[1],bins=30,color='#e6550d');axes[0,1].set_title('All initial-prefix signed heading (censored if failed)');axes[0,1].set_xlabel('degrees')
 axes[1,0].hist(duration,bins=30,color='#31a354');axes[1,0].set_title('Initial valid exposure duration');axes[1,0].set_xlabel('seconds')
 scatter=axes[1,1].scatter(initial,h[1],c=duration,s=6,cmap='viridis');fig.colorbar(scatter,ax=axes[1,1],label='initial-prefix valid seconds');axes[1,1].set_xlabel('reset heading (deg)');axes[1,1].set_ylabel('prefix heading drift (deg)')
 fig.suptitle('No policy: fixed default joint-position PD targets,1024env,one evaluation seed\nNo training-seed CI. Falling-prefix Euler yaw is not an unconditional20s heading estimate.')
 save(fig,'no_policy_heading_and_exposure')
 state=load('reconstruction_state.json')
 with np.load(state['trajectory']) as z:values=z['values'];valid=z['flags'][:,0].astype(bool)
 xy=values[:,:2];q=values[:,3:7].astype(float);yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2));previous=np.r_[state['heading_start_yaw'],yaw[:-1]]
 drift=np.cumsum(np.where(valid,np.arctan2(np.sin(yaw-previous),np.cos(yaw-previous)),0))*180/np.pi;t=np.arange(1,1001)*.02
 fig,axes=plt.subplots(1,2,figsize=(15,6));shown=xy.copy();shown[~valid]=np.nan
 axes[0].plot(shown[:,0],shown[:,1]);axes[0].scatter(*xy[0],c='green',label='first post-step');axes[0].scatter(*xy[-1],c='red',marker='x',label='20s')
 for i in range(0,1000,100):
  if valid[i]:axes[0].arrow(xy[i,0],xy[i,1],.12*np.cos(yaw[i]),.12*np.sin(yaw[i]),head_width=.04,color='#e6550d',length_includes_head=True)
 axes[0].set_aspect('equal',adjustable='datalim');axes[0].set_xlabel('world-axis x relative env origin (m)');axes[0].set_ylabel('world-axis y relative env origin (m)');axes[0].legend();axes[0].grid(alpha=.2)
 axes[1].plot(t,drift);axes[1].axhline(0,color='black',lw=.7);axes[1].set_xlabel('time (s)');axes[1].set_ylabel('reset-excluded signed heading drift (deg)');axes[1].grid(alpha=.2)
 fig.suptitle(f"Original control0.5: nearest pooled median ABS heading, all15360initial trials\ntrain seed{state['training_seed']} eval seed{state['eval_seed']} env{state['env_id']} | trial={state['heading_abs_degrees']:.3f}deg, median={state['group_median_absolute_heading_degrees']:.3f}deg; no survival filter")
 save(fig,'representative_world_heading')
 inputs=['lineage_heading_gate.json','mirror_comparisons.json','usd_symmetry_audit.json','no_policy_statistics.json','no_policy/default_eval1.json','reconstruction_state.json']
 paths={str(ROOT/n):sha(ROOT/n) for n in inputs}
 for p in [folder/'stand_yaw.npz',Path(state['trajectory'])]:paths[str(p)]=sha(p)
 binding=dict(source_sha256=SOURCE_SHA,sources={str(Path(__file__).resolve()):SOURCE_SHA},inputs=paths,outputs={p:sha(ROOT/p) for p in outputs},plot_count=7)
 (ROOT/'figure_outputs_sha256.json').write_text(json.dumps(binding,indent=2)+'\n')
 print('ISSUE16_FIGURES_COMPLETE7',flush=True)
if __name__=='__main__':main()
