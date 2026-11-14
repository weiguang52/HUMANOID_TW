"""All trace quantities on logarithmic axes; five seeds individually visible."""
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from issue21_common import ROOT,save,sha
from issue21_analysis import GROUPS

def main():
 pairs=json.loads((ROOT/'pair_analysis.json').read_text());manifest=[];figs=ROOT/'figures';figs.mkdir(exist_ok=True)
 quantities=list(GROUPS)+['action','root_position','root_quaternion','root_linear_velocity','root_angular_velocity','joint_pos','joint_vel','applied_torque','contact']
 for n in [16,1]:
  for cmd in ['stand','forward_05']:
   chosen=[x for x in pairs if x['kind']=='closed' and x['num_envs']==n and x['command']==cmd]
   if not chosen:continue
   fig,axs=plt.subplots(4,4,figsize=(20,14),constrained_layout=True)
   for ax,name in zip(axs.flat,quantities):
    for x in chosen:
     z=np.load(x['residual_file']);v=z[name+'_linf'].max(1);t=np.arange(len(v))+(0 if name in GROUPS or name=='action' else 1);ax.plot(t,np.maximum(v,1e-12),label='seed%d'%x['training_seed'],linewidth=1)
    ax.set_yscale('log');ax.set_title(name);ax.set_xlabel('Control step (20ms)');ax.set_ylabel('max abs component error');ax.grid(alpha=.2)
    for threshold in [1e-6,1e-4,1e-2]:ax.axhline(threshold,color='gray',alpha=.4,ls=':',lw=.8)
   axs.flat[0].legend(fontsize=8);fig.suptitle('Issue21 closed-loop mirrored residual | %s | num_envs=%d | native units (rad,m,N,Nm)'%(cmd,n));p=figs/('closed_%s_n%d.png'%(cmd,n));fig.savefig(p,dpi=120);plt.close(fig);manifest.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 for n in [16,1]:
  for cmd in ['stand','forward_05']:
   chosen=[x for x in pairs if x['kind']=='open' and x['num_envs']==n and x['command']==cmd]
   if not chosen:continue
   fig,axs=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
   for row,name in enumerate(['joint_pos','joint_vel','applied_torque']):
    for col,mode in enumerate(['air','ground']):
     ax=axs[row,col]
     for x in chosen:
      if x['mode']!=mode:continue
      z=np.load(x['residual_file']);v=z[name+'_linf'].max(1);ax.plot(np.arange(1,len(v)+1),np.maximum(v,1e-12),label='seed%d'%x['training_seed'])
     ax.set_yscale('log');ax.set_title(mode+' '+name);ax.set_xlabel('Post-step');ax.set_ylabel('max abs component error');ax.axhline(1e-5,ls=':',color='black',label='1e-5');ax.grid(alpha=.2)
   axs[0,0].legend(fontsize=8);fig.suptitle('Issue21 open-loop identical action pairs | %s | num_envs=%d'%(cmd,n));p=figs/('open_%s_n%d.png'%(cmd,n));fig.savefig(p,dpi=120);plt.close(fig);manifest.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 for cmd in ['stand','forward_05']:
  fig,axs=plt.subplots(3,2,figsize=(14,12),constrained_layout=True);present=False
  for col,n in enumerate([16,1]):
   chosen=[x for x in pairs if x['kind']=='closed' and x['num_envs']==n and x['command']==cmd]
   for row,name in enumerate(['joint_target','estimated_pd_torque','explicit_actuation_force']):
    ax=axs[row,col]
    for x in chosen:
     z=np.load(x['residual_file']);v=z[name+'_linf'].max(1);ax.plot(np.arange(1,len(v)+1),np.maximum(v,1e-12),label='seed%d'%x['training_seed']);present=True
    ax.set_yscale('log');ax.set_title('%s | num_envs%d'%(name,n));ax.set_xlabel('Post-step');ax.set_ylabel('max abs component error');ax.grid(alpha=.2)
  axs[0,0].legend(fontsize=8);fig.suptitle('Auxiliary actuator observables (PD estimate is not measured solver drive torque) | '+cmd)
  if present:
   p=figs/('actuator_aux_'+cmd+'.png');fig.savefig(p,dpi=120);manifest.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
  plt.close(fig)
 perturb=json.loads((ROOT/'perturbation_analysis.json').read_text())
 if perturb:
  fig,axs=plt.subplots(1,5,figsize=(20,5),constrained_layout=True)
  for ts,ax in enumerate(axs,1):
   rows=sorted([x for x in perturb if x['training_seed']==ts],key=lambda x:x['epsilon']);eps=[x['epsilon'] for x in rows]
   ax.violinplot([np.asarray(x['slopes'],dtype=float)[np.isfinite(x['slopes'])] for x in rows],positions=np.arange(len(rows)),showmeans=True,showextrema=True);ax.set_xticks(np.arange(len(rows)),['%+.2f'%e for e in eps],rotation=45);ax.axhline(0,color='black',lw=.8);ax.set_title('Training seed%d'%ts);ax.set_xlabel('Signed yaw velocity epsilon rad/s');ax.set_ylabel('5-20s OLS heading slope deg/s');ax.grid(alpha=.2)
  p=figs/'perturbation_slopes.png';fig.savefig(p,dpi=120);plt.close(fig);manifest.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 save('figure_manifest.json',manifest);print('ISSUE21_PLOTS',len(manifest),flush=True)
if __name__=='__main__':main()
