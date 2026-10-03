"""Fixed-camera original-speed visualization of HumanML3D new_joints."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
p.add_argument('--fps',type=float,default=20)
a=p.parse_args();x=np.load(a.source,allow_pickle=False)
assert x.ndim==3 and x.shape[1:]==(22,3) and np.isfinite(x).all()
assert a.fps>0
# HumanML has Y up; display Z up, preserving all positions and temporal samples.
y=x[:,:,[0,2,1]]
chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]
colors=['#ef8a23','#2788ce','#657381','#ef8a23','#2788ce']
fig=plt.figure(figsize=(12.8,7.2),facecolor='#fafbfc')
axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
lo=y.min(axis=(0,1));hi=y.max(axis=(0,1));center=(lo+hi)/2
span=max(float((hi-lo).max()),1)*1.15
lines=[]
for ax,az,title in zip(axes,[-65,25],['Oblique view','Alternate view']):
 ax.set_xlim(center[0]-span/2,center[0]+span/2)
 ax.set_ylim(center[1]-span/2,center[1]+span/2)
 ax.set_zlim(min(0,lo[2])-.05,min(0,lo[2])-.05+span)
 ax.set_box_aspect((1,1,1));ax.view_init(elev=12,azim=az)
 ax.set_xlabel('X (m)');ax.set_ylabel('Z (m)');ax.set_zlabel('Height (m)')
 ax.set_title(title);ax.grid(alpha=.2)
 for chain,c in zip(chains,colors):
  line,=ax.plot([],[],[],color=c,lw=3,marker='o',markersize=3)
  lines.append((line,chain))
fig.suptitle('HumanML3D '+a.source.stem+' | Original 22-joint skeleton',fontsize=17)
time=fig.text(.5,.08,'',ha='center',fontsize=12)
fig.text(.5,.035,'Blue: left limbs | Orange: right limbs | Fixed cameras | 1x source speed',ha='center',fontsize=11)
fig.subplots_adjust(left=.02,right=.97,bottom=.12,top=.88,wspace=.06)
a.output.parent.mkdir(parents=True,exist_ok=True)
writer=FFMpegWriter(fps=a.fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','20','-movflags','+faststart'])
with writer.saving(fig,str(a.output),dpi=100):
 for i,frame in enumerate(y):
  for line,chain in lines:
   q=frame[chain];line.set_data(q[:,0],q[:,1]);line.set_3d_properties(q[:,2])
  time.set_text(f'{i/a.fps:.2f} s | Frame {i+1}/{len(x)} | {a.fps:g} fps')
  writer.grab_frame()
  if i==len(x)//2:fig.savefig(a.output.with_suffix('.png'),dpi=100)
plt.close(fig)
a.output.with_suffix('.json').write_text(json.dumps(dict(source=str(a.source),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),shape=list(x.shape),fps=a.fps,duration_s=len(x)/a.fps,coordinate_display='source XYZ -> XZY (Y up)',processing='No smoothing, retargeting, interpolation or playback slowdown'),indent=2))
print(a.output,flush=True)
