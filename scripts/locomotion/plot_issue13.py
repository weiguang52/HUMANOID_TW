"""Publication/export figures from the already frozen issue13 comparisons."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from issue13_common import ROOT,FORWARD,CLEAN
from issue13_tracking import SCOPES
def main():
 assert (ROOT/"analysis_complete").exists()
 comp=json.loads((ROOT/"paired_comparisons.json").read_text());gates=json.loads((ROOT/"primary_gate_comparisons.json").read_text())
 decision=json.loads((ROOT/"decision.json").read_text());dest=ROOT/"figures";dest.mkdir(exist_ok=True)
 plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":.2})
 colors={"control":"#2878A5","treatment":"#E48A26"};names={"forward_03":"0.3","forward_05":"0.5","forward_08":"0.8","yaw_positive":"yaw+0.3","yaw_negative":"yaw-0.3"}
 def get(condition,cmd,part,metric,scope=SCOPES[0]):return comp[":".join([condition,cmd,scope,part,metric])]
 def point(ax,x,s,color,label=None):
  mean=s["mean"];lo,hi=s["bootstrap95"];ax.vlines(x,lo,hi,color=color,lw=1.7);ax.plot(x,mean,"o",color=color,label=label)
 def save(fig,name):
  fig.tight_layout();fig.savefig(dest/(name+".png"),dpi=150);fig.savefig(dest/(name+".svg"))
  p=dest/(name+".svg");p.write_text("\n".join(x.rstrip() for x in p.read_text().splitlines()).rstrip()+"\n");plt.close(fig)
 display=FORWARD+["yaw_positive","yaw_negative"]
 fig,axes=plt.subplots(1,2,figsize=(13,4.5))
 for j,metric in enumerate(["xy_error","yaw_error"]):
  ax=axes[j]
  for i,command in enumerate(display):
   d=get("clean",command,"all",metric)
   for group,offset in [("control",-.12),("treatment",.12)]:point(ax,i+offset,d[group],colors[group],group if i==0 else None)
  ax.set_xticks(range(5),[names[c] for c in display]);ax.set_ylabel("xy error (m/s)" if j==0 else "yaw error (rad/s)");ax.set_ylim(bottom=0);ax.legend()
 fig.suptitle("Clean nominal condition; five training-seed means and seed-bootstrap95%")
 save(fig,"clean_tracking")
 fig,axes=plt.subplots(2,3,figsize=(14,8))
 for col,command in enumerate(FORWARD):
  for row,metric in enumerate(["xy_error","yaw_error"]):
   ax=axes[row,col]
   for i,part in enumerate(["all","inside","outside"]):
    d=get("combined",command,part,metric)
    for group,offset in [("control",-.12),("treatment",.12)]:point(ax,i+offset,d[group],colors[group],group if i==0 else None)
   ax.set_xticks(range(3),["all","inside1s","outside"]);ax.set_title(names[command]+" m/s");ax.set_ylim(bottom=0)
   if col==0:ax.set_ylabel("xy error (m/s)" if row==0 else "yaw error (rad/s)");ax.legend()
 fig.suptitle("Bounded combined: conditional window means; paired seed-bootstrap95%")
 save(fig,"window_tracking")
 fig,ax=plt.subplots(figsize=(14,9));labels=[]
 for i,(key,d) in enumerate(gates.items()):
  condition,command,scope,part,metric=key.split(":");labels.append(condition+" | "+names[command]+" | "+("out " if part=="outside" else "")+metric)
  if d["relative_change"] is None:ax.text(0,i,"ratio undefined");continue
  ci=d["relative_change_bootstrap95"];color="#B33B32" if d["point_worse_than5percent_and_positive95"] else "#2878A5"
  if ci is not None:ax.hlines(i,100*ci[0],100*ci[1],color=color,lw=1.7)
  ax.plot(100*d["relative_change"],i,"o",color=color)
 ax.axvline(0,color="gray",ls=":");ax.axvline(5,color="#B33B32",ls="--",label="5% bound");ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
 ax.set_xlabel("Treatment/control group-mean error change (%)");ax.legend();ax.set_title("Frozen13 comparisons; five paired training seeds;95% intervals unadjusted\n"+decision["decision"])
 save(fig,"frozen_gate_contrasts")
 fig,axes=plt.subplots(2,3,figsize=(14,8))
 for col,(condition,part,title) in enumerate([("clean","all","Clean"),("combined","inside","Combined inside1s"),("combined","outside","Combined outside")]):
  for row,metric in enumerate(["vx_bias","vx_temporal_sd_mean"]):
   ax=axes[row,col]
   for i,command in enumerate(FORWARD):
    d=get(condition,command,part,metric)
    for group,offset in [("control",-.1),("treatment",.1)]:point(ax,i+offset,d[group],colors[group],group if i==0 else None)
   ax.set_xticks(range(3),[names[c] for c in FORWARD]);ax.set_title(title);ax.set_xlabel("Forward command (m/s)")
   if row==0:ax.axhline(0,color="gray",ls=":")
   else:ax.set_ylim(bottom=0)
   if col==0:ax.set_ylabel("Mean body vx minus command (m/s)" if row==0 else "Mean per-trial temporal SD vx (m/s)");ax.legend()
 fig.suptitle("Systematic forward bias versus temporal fluctuation; seed-bootstrap95%")
 save(fig,"bias_and_variability")
 fig,axes=plt.subplots(1,3,figsize=(13,4.5))
 for ax,command in zip(axes,FORWARD):
  for i,group in enumerate(["control","treatment"]):
   inside=get("combined",command,"inside","xy_weighted_contribution")[group]["mean"]
   outside=get("combined",command,"outside","xy_weighted_contribution")[group]["mean"]
   overall=get("combined",command,"all","xy_error")[group]["mean"]
   assert abs(inside+outside-overall)<1e-10
   ax.bar(i,inside,color="#E48A26",label="inside contribution" if i==0 else None)
   ax.bar(i,outside,bottom=inside,color="#2878A5",label="outside contribution" if i==0 else None)
   ax.text(i,overall,f"{overall:.4f}",ha="center",va="bottom")
  ax.set_xticks([0,1],["control","treatment"]);ax.set_title(names[command]+" m/s");ax.set_ylim(bottom=0)
 axes[0].set_ylabel("Full-horizon xy error contribution (m/s)");axes[0].legend()
 fig.suptitle("Exposure-weighted inside+outside contributions equal original overall error\nDescriptive five-training-seed means; intervals remain in paired tables")
 save(fig,"weighted_error_contributions")
 fig,ax=plt.subplots(figsize=(11,5))
 variants=[("outside",SCOPES[0],"post-event primary","o",-.15),("reward_aligned_outside",SCOPES[0],"reward-boundary sensitivity","s",0),("outside",SCOPES[1],"initial-episode sensitivity","^",.15)]
 for part,scope,label,marker,offset in variants:
  for i,command in enumerate(FORWARD):
   d=get("combined",command,part,"xy_error",scope);ci=d["relative_change_bootstrap95"]
   if d["relative_change"] is None:continue
   x=i+offset;color={"o":"#2878A5","s":"#E48A26","^":"#5B9756"}[marker]
   if ci is not None:ax.vlines(x,100*ci[0],100*ci[1],color=color)
   ax.plot(x,100*d["relative_change"],marker,color=color,label=label if i==0 else None)
 ax.axhline(5,color="#B33B32",ls="--",label="5% bound");ax.axhline(0,color="gray",ls=":")
 ax.set_xticks(range(3),[names[c] for c in FORWARD]);ax.set_xlabel("Forward command (m/s)");ax.set_ylabel("Outside xy mean degradation (%)");ax.legend()
 ax.set_title("Prespecified sampling sensitivities; primary rule is never reselected")
 save(fig,"window_scope_sensitivity")
 assert len(list(dest.glob("*.png")))==6 and len(list(dest.glob("*.svg")))==6
 (ROOT/"plots_complete").write_text("Six PNG+SVG scientific figures from frozen comparisons.\n");print("ISSUE13_PLOTS_COMPLETE",flush=True)
if __name__=="__main__":main()
