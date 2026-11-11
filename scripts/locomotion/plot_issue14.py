"""Six remote scientific figures for the three-group frozen issue14 report."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from issue14_common import ROOT,sha
from issue13_tracking import SCOPES
GROUPS=["control","treatment","clock"];COLORS=["#26749b","#df8928","#339266"];SPEEDS=[.3,.5,.8];CMDS=["forward_03","forward_05","forward_08"]
def main():
 assert (ROOT/"analysis_complete").exists();out=ROOT/"figures";out.mkdir(exist_ok=True)
 comp=json.loads((ROOT/"paired_comparisons.json").read_text());falls=json.loads((ROOT/"fall_results.json").read_text());decision=json.loads((ROOT/"decision.json").read_text())
 plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.2})
 def get(pair,cond,cmd,part,metric):return comp[":".join([pair,cond,cmd,SCOPES[0],part,metric])]
 def stat(g,cond,cmd,part,metric):
  pair="clock_vs_treatment" if g=="treatment" else "clock_vs_control";d=get(pair,cond,cmd,part,metric)
  return d["treatment" if g=="clock" else "control"]
 def err(s):return np.array([[s["mean"]-s["bootstrap95"][0]],[s["bootstrap95"][1]-s["mean"]]])
 def save(fig,name):
  fig.tight_layout();fig.savefig(out/(name+".png"),dpi=150);fig.savefig(out/(name+".svg"));p=out/(name+".svg");p.write_text("\n".join(x.rstrip() for x in p.read_text().splitlines()).rstrip()+"\n");plt.close(fig)
 clean=["forward_03","yaw_positive","yaw_negative","forward_05","forward_08"];labels=["0.3","yaw+0.3","yaw-0.3","0.5","0.8"]
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for ax,metric,unit in zip(axes,["xy_error","yaw_error"],["m/s","rad/s"]):
  for gi,(g,color) in enumerate(zip(GROUPS,COLORS)):
   for i,cmd in enumerate(clean):
    s=stat(g,"clean",cmd,"all",metric);ax.errorbar(i+(gi-1)*.19,s["mean"],yerr=err(s),fmt="o",color=color,capsize=3,label=g if i==0 else None)
  ax.set_xticks(range(5),labels);ax.set_ylabel(metric+" ("+unit+")");ax.set_xlabel("Clean separate command");ax.legend()
 fig.suptitle("Clean tracking; five training seeds, bootstrap95% unadjusted");save(fig,"clean_tracking")
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 for row,metric in enumerate(["xy_error","yaw_error"]):
  for col,cmd in enumerate(CMDS):
   ax=axes[row,col]
   for gi,(g,color) in enumerate(zip(GROUPS,COLORS)):
    for i,part in enumerate(["inside","outside"]):
     s=stat(g,"combined",cmd,part,metric);ax.errorbar(i+(gi-1)*.18,s["mean"],yerr=err(s),fmt="o",color=color,capsize=3,label=g if i==0 else None)
   ax.set_xticks([0,1],["inside1s","outside"]);ax.set_ylabel(metric+(" (m/s)" if row==0 else " (rad/s)"));ax.set_title(str(SPEEDS[col])+"m/s")
   if row==0 and col==0:ax.legend()
 fig.suptitle("Combined conditional tracking; original full-horizon nonterminal samples");save(fig,"window_tracking")
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 conditions=[("clean","all","Clean"),("combined","inside","Combined inside1s"),("combined","outside","Combined outside")]
 for col,(cond,part,title) in enumerate(conditions):
  for row,metric in enumerate(["vx_bias","vx_temporal_sd_mean"]):
   ax=axes[row,col]
   for gi,(g,color) in enumerate(zip(GROUPS,COLORS)):
    for i,cmd in enumerate(CMDS):
     s=stat(g,cond,cmd,part,metric);ax.errorbar(SPEEDS[i]+(gi-1)*.014,s["mean"],yerr=err(s),fmt="o",color=color,capsize=3,label=g if i==0 else None)
   if row==0:ax.axhline(0,color="gray",ls=":")
   ax.set_title(title);ax.set_xlabel("Forward command (m/s)");ax.set_ylabel("Mean vx-command (m/s)" if row==0 else "Mean trial temporal SD vx (m/s)")
   if row==1 and col==0:ax.legend()
 fig.suptitle("Systematic forward bias and within-trial temporal fluctuation; unadjusted seed95%");save(fig,"bias_and_variability")
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 primary=decision["primary"]["paired_difference"];ax=axes[0,0];ax.errorbar([0],[primary["mean"]],yerr=np.array([[primary["mean"]-primary["bootstrap95"][0]],[primary["bootstrap95"][1]-primary["mean"]]]),fmt="o",capsize=5)
 ax.axhline(0,color="red",ls="--");ax.set_xticks([0],["clock - treatment"]);ax.set_ylabel("Clean0.8 xy difference (m/s)");ax.set_title("Primary: entire95% below0 required")
 ax=axes[0,1]
 for i,cmd in enumerate(CMDS[1:]):
  d=get("clock_vs_control","clean",cmd,"all","xy_error");ci=d["relative_change_bootstrap95"];mean=100*d["relative_change"]
  ax.errorbar(i,mean,yerr=np.array([[mean-100*ci[0]],[100*ci[1]-mean]]),fmt="o",capsize=4,color=COLORS[2])
 ax.axhline(5,color="red",ls="--",label="5% point bound");ax.set_xticks([0,1],["0.5","0.8"]);ax.set_ylabel("Clock/control xy change (%)");ax.set_title("Clean point bounds");ax.legend()
 ax=axes[0,2];s=decision["clean08_vx"]["paired_difference"];ax.errorbar([0],[s["mean"]],yerr=np.array([[s["mean"]-s["bootstrap95"][0]],[s["bootstrap95"][1]-s["mean"]]]),fmt="o",capsize=5)
 ax.axhline(0,color="red",ls="--");ax.set_xticks([0],["clock - control"]);ax.set_ylabel("Clean0.8 mean vx difference (m/s)");ax.set_title("95% must not be entirely negative")
 for col,cmd in enumerate(CMDS[:2]):
  ax=axes[1,col]
  for i,(g,color) in enumerate(zip(GROUPS,COLORS)):
   s=falls["combined:"+cmd]["groups"][g]["seed_summary"];ax.errorbar(i,100*s["mean"],yerr=100*err(s),fmt="o",color=color,capsize=4)
  if col==0:ax.axhline(1,color="red",ls="--",label="1% upper bound")
  else:ax.axhline(100*falls["combined:"+cmd]["groups"]["control"]["seed_summary"]["mean"],color=COLORS[0],ls="--",label="control point mean")
  ax.set_xticks(range(3),GROUPS);ax.set_ylabel("Combined first-fall rate (%)");ax.set_title(str(SPEEDS[col])+"m/s");ax.legend()
 ax=axes[1,2];ax.axis("off");texts=["Frozen criteria","Primary mitigation: "+str(decision["primary_mitigation_supported"]),"All gates pass: "+str(decision["all_gates_pass"]),"Formal: five x1500updates","No weight/encoding/budget search","Multiple comparisons unadjusted"]
 ax.text(.02,.9,"\n\n".join(texts),va="top")
 fig.suptitle("Prespecified primary and complete acceptance gates; no criterion reselection");save(fig,"primary_and_acceptance")
 table=list(csv.DictReader((ROOT/"actual_speed_fall_table.csv").open()));fig,axes=plt.subplots(1,3,figsize=(15,5))
 for ax,cmd,speed in zip(axes,CMDS,SPEEDS):
  for g,color in zip(GROUPS,COLORS):
   x=next(t for t in table if t["condition"]=="combined" and t["command"]==cmd and t["group"]==g);vx=float(x["mean_actual_vx"]);f=100*float(x["fall_seed_mean"])
   ax.errorbar(vx,f,xerr=[[vx-float(x["actual_vx95_low"])] ,[float(x["actual_vx95_high"])-vx]],
    yerr=[[f-100*float(x["fall95_low"])],[100*float(x["fall95_high"])-f]],fmt="o",color=color,label=g,capsize=3)
  ax.axvline(speed,color="gray",ls=":");ax.set_title("Command "+str(speed)+"m/s");ax.set_xlabel("Measured mean body vx (m/s)");ax.set_ylabel("Initial-trial first falls (%)")
 axes[0].legend();fig.suptitle("Achieved speed beside falls; marginal seed95% intervals, descriptive not causal");save(fig,"actual_speed_vs_falls")
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 for col,cmd in enumerate(CMDS):
  bins=json.loads((ROOT/("actual_speed_bins_"+cmd+".json")).read_text())["bins"]
  for gi,(g,color) in enumerate(zip(GROUPS,COLORS)):
   for b in bins:
    v=b["groups"][g];x=(b["bin_id"]-.5)*.05
    axes[1,col].bar(x+(gi-1)*.012,v["n"],width=.011,color=color,label=g if b["bin_id"]==0 else None)
    if b["common_all3_groups"] and v["n"]:
     s=v["seed_summary"]
     if s is not None:axes[0,col].errorbar(x+(gi-1)*.004,100*s["mean"],yerr=100*err(s),fmt="o",markersize=3,color=color,capsize=2)
     else:axes[0,col].scatter([x+(gi-1)*.004],[100*v["pooled_rate"]],facecolors="none",edgecolors=color,s=18)
  for row in range(2):
   axes[row,col].set_title(str(SPEEDS[col])+"m/s command")
   axes[row,col].set_xlabel("Achieved-vx bin (m/s); tails unbounded")
   axes[row,col].set_xticks([-.025,.025,.225,.425,.625,.825,1.025],["<0","0-.05",".20-.25",".40-.45",".60-.65",".80-.85",">=1"],rotation=30,fontsize=8)
  axes[0,col].set_ylabel("Fall rate in common bins (%)");axes[1,col].set_ylabel("Initial-trial count per group")
 axes[1,0].legend();fig.suptitle("Post-treatment speed bins: filled=all5seed mean/95%; open=pooled sparse support; no causal adjustment",fontsize=11);save(fig,"actual_speed_common_bins")
 inputs=["paired_comparisons.json","fall_results.json","decision.json","actual_speed_fall_table.csv"]+["actual_speed_bins_"+c+".json" for c in CMDS]
 (ROOT/"figure_outputs_sha256.json").write_text(json.dumps({"source_sha256":{str(Path(__file__)):sha(Path(__file__))},"input_sha256":{name:sha(ROOT/name) for name in inputs},"figures_sha256":{str(p.relative_to(ROOT)):sha(p) for p in out.iterdir() if p.suffix in {".png",".svg"}},"derived_code_not_claimed_prospectively_frozen":True},indent=2))
 (ROOT/"plots_complete").write_text("Six PNG/SVG figures; manual visual review still required.\n");print("ISSUE14_PLOTS_COMPLETE",flush=True)
if __name__=="__main__":main()
