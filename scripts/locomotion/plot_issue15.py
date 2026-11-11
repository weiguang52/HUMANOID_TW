"""Plot complete issue15 lineage without changing the frozen decision definitions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from issue15_common import ROOT,sha
FORWARD=["forward_03","forward_05","forward_08"]
COLORS=["#225ea8","#1d91c0","#41b6c4","#7fcdbb","#c7e9b4","#fed976","#feb24c","#fd8d3c","#f03b20","#bd0026","#6a51a3"]
def main():
 assert (ROOT/"analysis_complete").exists()
 node=json.loads((ROOT/"node_statistics.json").read_text());cohort=json.loads((ROOT/"cohort.json").read_text());order=cohort["display_order"];x=np.arange(len(order));out=ROOT/"figures";out.mkdir(exist_ok=True);outputs=[]
 def metric(name,cmd,key):return node[name][cmd]["metrics"].get(key)
 def series(ax,cmd,key,label,color="#225ea8",legendlabel=None):
  means=[];low=[];high=[]
  for n in order:
   s=metric(n,cmd,key)
   if s is None:means.append(np.nan);low.append(0);high.append(0);continue
   means.append(s["mean"]);ci=s["bootstrap95"];low.append(s["mean"]-ci[0] if ci else 0);high.append(ci[1]-s["mean"] if ci else 0)
  ax.errorbar(x,means,yerr=[low,high],fmt="o-",lw=1,capsize=2,color=color,label=legendlabel)
  for i,n in enumerate(order):
   if node[n][cmd]["training_seeds"]==1 and np.isfinite(means[i]):ax.plot(i,means[i],"s",mfc="white",mec=color)
  ax.set_xticks(x);ax.set_xticklabels([n+"\nn="+str(node[n][cmd]["training_seeds"]) for n in order],rotation=65,ha="right",fontsize=7);ax.set_ylabel(label);ax.grid(alpha=.2)
 def finish(fig,name):
  fig.tight_layout();fig.savefig(out/(name+".png"),dpi=150);fig.savefig(out/(name+".svg"));plt.close(fig)
  outputs.extend([str((out/(name+s)).relative_to(ROOT)) for s in [".png",".svg"]])
 commands=["stand",*FORWARD,"yaw_positive","yaw_negative"]
 fig,axes=plt.subplots(3,6,figsize=(24,13))
 for j,cmd in enumerate(commands):
  for i,(key,label) in enumerate([("full_yaw_error","yaw MAE (rad/s)"),("full_xy_error","xy error (m/s)"),("fall_rate","first-fall fraction")]):
   series(axes[i,j],cmd,key,label);axes[i,j].set_title(cmd)
 fig.suptitle("Unified clean lineage; training-seed 95% bootstrap, unadjusted; open squares: one training seed (no CI)\nDisplay order includes sibling comparisons and identical-model cohort expansion; see actual ancestry.",y=1.01)
 finish(fig,"lineage_yaw_xy_falls")
 fig,axes=plt.subplots(3,4,figsize=(22,13))
 for i,cmd in enumerate(FORWARD):
  for j,(key,label) in enumerate([("full_yaw_rms_trial_bias","RMS per-trial signed bias (rad/s)"),("full_yaw_mean_trial_temporal_sd","Mean per-trial temporal SD (rad/s)"),("full_yaw_within_energy_fraction","Within-trial yaw MSE fraction"),("full20s_survivor_heading_mean_abs_degrees","20s survivor mean |heading drift| (deg)")]):
   series(axes[i,j],cmd,key,label);axes[i,j].set_title(cmd)
   if j==2:axes[i,j].axhline(.5,color="black",ls="--",lw=.8)
 fig.suptitle("Curving versus oscillation without cross-trial sign cancellation; Euler heading excludes reset jumps\n20s heading uses no-reset survivors only; counts/falls and all initial-prefix drifts remain in tables.",y=1.01)
 finish(fig,"yaw_decomposition_heading")
 comparisons=json.loads((ROOT/"lineage_comparisons.json").read_text())
 fig,axes=plt.subplots(1,2,figsize=(18,8))
 matrix=np.asarray([[100*next(c["relative_increase"] for c in comparisons if c["previous"]==a and c["current"]==b and c["command"]==cmd) for cmd in FORWARD] for a,b in cohort["comparison_edges"]])
 vmax=max(30,float(np.nanmax(abs(matrix))))
 im=axes[0].imshow(matrix,cmap="coolwarm",vmin=-vmax,vmax=vmax,aspect="auto")
 axes[0].set_yticks(range(len(matrix)));axes[0].set_yticklabels([a+" -> "+b for a,b in cohort["comparison_edges"]],fontsize=8)
 axes[0].set_xticks(range(3));axes[0].set_xticklabels(FORWARD)
 for i,(a,b) in enumerate(cohort["comparison_edges"]):
  for j,cmd in enumerate(FORWARD):
   c=next(c for c in comparisons if c["previous"]==a and c["current"]==b and c["command"]==cmd)
   suffix=" *" if c["meets_issue15_descriptive_or_multiseed_criterion"] else ""
   if c["single_seed_descriptive_only"]:suffix+=" 1seed"
   if c["alias_cohort_expansion_not_configuration_effect"]:suffix=" alias"
   axes[0].text(j,i,f"{matrix[i,j]:+.1f}%"+suffix,ha="center",va="center",fontsize=8)
 fig.colorbar(im,ax=axes[0],label="Full-node yaw MAE change (%)")
 axes[0].set_title(">20% plus positive CI for multi-seed later nodes; * marks frozen criterion")
 axes[1].axis("off");axes[1].text(0,1,"Read comparisons with their scope:\n\n"
 "Single-seed nodes: descriptive threshold only.\n"
 "Single -> multi: bootstrap later seeds against\na fixed earlier policy, conditional inference only.\n\n"
 "Multi -> multi: matched seed intersection;\nunequal full cohorts are disclosed.\n\n"
 "bounded -> requested1500 includes the same\nseed1/2 checkpoint bytes; matched change zero.\n\n"
 "FT5 and yaw_s1 share FT4; robust05 follows\nyaw_s1, not FT5.\n"
 "Strengthen and bounded restart from robust05.\n"
 "Control, treatment and clock share requested4500\nparents; display edges are not all parent-child.\n\n"
 "Configuration differences and training randomness\nmust be checked before causal attribution.\n"
 "All comparisons unadjusted.",va="top",fontsize=11)
 finish(fig,"lineage20percent_comparisons")
 fig,axes=plt.subplots(3,2,figsize=(16,13))
 for i,cmd in enumerate(FORWARD):
  series(axes[i,0],cmd,"yaw_peak_hz_mean","Mean residual yaw peak (Hz)",legendlabel="planned0.2..10Hz")
  series(axes[i,0],cmd,"fullband_yaw_peak_hz_mean","Mean residual yaw peak (Hz)",color="#f03b20",legendlabel="all positive..25Hz supplement")
  series(axes[i,1],cmd,"peak_over_step_mean","Mean yaw peak / footstrike step frequency",legendlabel="planned0.2..10Hz")
  series(axes[i,1],cmd,"fullband_peak_over_step_mean","Mean yaw peak / footstrike step frequency",color="#f03b20",legendlabel="all positive..25Hz supplement")
  axes[i,0].legend(fontsize=7);axes[i,1].legend(fontsize=7)
  axes[i,0].set_title(cmd);axes[i,1].set_title(cmd);axes[i,1].axhline(1,color="black",ls="--",lw=.8)
 fig.suptitle("Initial uninterrupted segments >=5s; mean removed + Hann FFT, planned band0.2..10Hz plus supplementary full observable positive..25Hz\nForce-contact touchdown counts debounced0.1s. Frequency coincidence is descriptive, not a causal mechanism.",y=1.01)
 finish(fig,"yaw_peak_and_step_frequency")
 legacy=json.loads((ROOT/"original_ft5_protocol_comparison.json").read_text())
 fig,axes=plt.subplots(1,2,figsize=(14,5))
 for es,e in enumerate(legacy):
  rows=e["rows"];k=np.arange(len(rows))
  axes[0].plot(k,[r["original_yaw"] for r in rows],"o",color=COLORS[es],label="original evalseed"+str(e["eval_seed"]))
  axes[0].plot(k,[r["rerun_yaw"] for r in rows],"+",color=COLORS[es],ms=10)
 axes[0].set_xticks(k);axes[0].set_xticklabels([r["command"] for r in rows],rotation=40,ha="right");axes[0].set_ylabel("yaw MAE (rad/s)");axes[0].legend(fontsize=7);axes[0].set_title("Original #1 protocol: dots archive, + rerun")
 for i,cmd in enumerate(FORWARD):
  s=metric("ft5",cmd,"full_yaw_error");axes[1].plot(i,s["mean"],"s",color="#225ea8",label="unified clean" if i==0 else None)
  if cmd=="forward_03":
   vals=[next(r for r in e["rows"] if r["command"]==cmd)["rerun_yaw"] for e in legacy]
   axes[1].plot(np.full(3,i),vals,"x",color="#f03b20",label="original protocol eval seeds")
 axes[1].set_xticks(range(3));axes[1].set_xticklabels(FORWARD);axes[1].set_ylabel("FT5 yaw MAE (rad/s)");axes[1].legend(fontsize=8);axes[1].set_title("Same FT5 policy; protocol and sample-size differences")
 fig.suptitle("FT5 has ONE training seed; three evaluation seeds are not independent training replications",y=1.01)
 finish(fig,"original_ft5_protocol_comparison")
 sources=[Path(__file__),Path(__file__).with_name("issue15_common.py")]
 inputs=["node_statistics.json","cohort.json","lineage_comparisons.json","original_ft5_protocol_comparison.json"]
 (ROOT/"figure_outputs_sha256.json").write_text(json.dumps({"sources":{str(p):sha(p) for p in sources},"inputs":{n:sha(ROOT/n) for n in inputs},"outputs":{n:sha(ROOT/n) for n in outputs}},indent=2)+"\n")
 (ROOT/"plots_complete").write_text("Five scientific PNG/SVG figures, manual review still required.\n")
 print("ISSUE15_PLOTS_COMPLETE",flush=True)
if __name__=="__main__":main()
