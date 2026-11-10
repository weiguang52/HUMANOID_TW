"""Scientific figures from complete issue12 tables; no selection or model changes."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from issue12_common import ROOT,COMMANDS

OUT=ROOT/"figures";OUT.mkdir(exist_ok=True)
COLORS={"control":"#2878b5","treatment":"#e07a20"}
SPEEDS=["0.3","0.5","0.8"]

def read(name):return json.loads((ROOT/name).read_text())
def finish(fig,name):
    fig.savefig(OUT/(name+".png"),dpi=150)
    fig.savefig(OUT/(name+".svg"))
    plt.close(fig)
def error(mean,ci,scale=1):
    return np.maximum(np.asarray([[mean-ci[0]],[ci[1]-mean]])*scale,0)
def rolling(x,n=50):return np.convolve(x,np.ones(n)/n,mode="valid")

def main():
    g=read("group_results.json");effects=read("paired_effects.json");metrics=read("tracking_comparison.json")
    plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.2})
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for i,(ax,cmd) in enumerate(zip(axes,COMMANDS)):
        c=g["control:"+cmd]["seed_summary"];t=g["treatment:"+cmd]["seed_summary"]
        for j in range(5):ax.plot([0,1],[100*c["per_seed"][j],100*t["per_seed"][j]],color=".65",lw=1)
        for x,(group,d) in enumerate([("control",c),("treatment",t)]):
            ax.scatter(np.full(5,x),np.asarray(d["per_seed"])*100,color=COLORS[group],alpha=.75,s=25)
            ax.errorbar(x,d["mean"]*100,yerr=error(d["mean"],d["bootstrap95"],100),color=COLORS[group],fmt="D",capsize=5,ms=8)
        ax.set_xticks([0,1],["control","treatment"]);ax.set_title(SPEEDS[i]+"m/s; paired5seeds")
        ax.set_ylabel("Fall rate (%)");ax.set_ylim(bottom=0)
        if i==0:ax.axhline(1,color="#b22222",ls="--",label="1% criterion");ax.legend(fontsize=8)
    fig.suptitle("6000 cumulative updates; bootstrap95% of five-seed means (panel-specific scales)")
    finish(fig,"fall_rates")

    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    for i,cmd in enumerate(COMMANDS):
        d=effects[cmd];mean=d["mean_difference"]
        ax.scatter(i+np.linspace(-.07,.07,5),np.asarray(d["per_seed_differences"])*100,s=25,color="#555555")
        ax.errorbar(i,mean*100,yerr=error(mean,d["bootstrap95"],100),fmt="D",color=COLORS["treatment"],capsize=6,ms=8)
    ax.axhline(0,color="black",ls="--");ax.set_xticks(range(3),[x+"m/s" for x in SPEEDS])
    ax.set_ylabel("Treatment minus control fall rate (percentage points)")
    ax.set_title("Paired training-seed bootstrap95%;3125 joint resamples")
    finish(fig,"paired_effects")

    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,key,unit in [(axes[0],"xy","m/s"),(axes[1],"yaw","rad/s")]:
        for j,group in enumerate(["control","treatment"]):
            for i,cmd in enumerate(COMMANDS):
                d=metrics[key][cmd][group];x=i+(j-.5)*.2
                ax.errorbar(x,d["mean"],yerr=error(d["mean"],d["bootstrap95"]),fmt="o",
                            color=COLORS[group],capsize=4,label=group if i==0 else None)
        ax.set_xticks(range(3),SPEEDS);ax.set_xlabel("Forward command(m/s)")
        ax.set_ylabel(key+" tracking error("+unit+")");ax.set_ylim(bottom=0);ax.legend()
    ax=axes[2]
    for i,cmd in enumerate(COMMANDS):
        d=metrics["xy"][cmd];ratio=d["ratio_of_seed_means"];ci=d["ratio_bootstrap95"]
        if ratio is None:ax.text(i,1,"N/A",ha="center")
        elif ci is None:ax.plot(i,ratio,"D",color=COLORS["treatment"])
        else:ax.errorbar(i,ratio,yerr=error(ratio,ci),fmt="D",color=COLORS["treatment"],capsize=4)
    ax.axhline(1,color="gray",ls=":");ax.axhline(1.1,color="#b22222",ls="--",label="10% bound")
    ax.set_xticks(range(3),SPEEDS);ax.set_ylabel("Treatment/control xy mean ratio");ax.legend()
    fig.suptitle("#7 non-terminal-frame metrics; later normal auto-reset episodes included")
    finish(fig,"tracking_errors")

    rows=list(csv.DictReader((ROOT/"reward_monitor_timeseries.csv").open()))
    series={}
    for seed in range(1,6):
        data=[r for r in rows if r["group"]=="treatment" and int(r["training_seed"])==seed]
        assert len(data)==1500
        series[seed]={k:np.asarray([float(r[k]) for r in data]) for k in
            ["new_reward_per_step","old_reward_per_step","xy_tracking_reward_per_step","yaw_tracking_reward_per_step"]}
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained");x=np.arange(50,1501)
    for key,label,color in [("new_reward_per_step","|new penalty|","#e07a20"),
            ("old_reward_per_step","old net reward","#555555"),
            ("xy_tracking_reward_per_step","xy tracking","#2878b5"),
            ("yaw_tracking_reward_per_step","yaw tracking","#3a923a")]:
        a=np.mean([series[s][key] for s in series],axis=0)
        if key=="new_reward_per_step":a=np.abs(a)
        axes[0].plot(x,rolling(a),label=label,color=color)
    for seed,d in series.items():
        numerator=rolling(abs(d["new_reward_per_step"]));denominator=abs(rolling(d["old_reward_per_step"]))
        ratio=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>1e-12)
        axes[1].plot(x,100*ratio,alpha=.55,label=f"seed{seed}")
    axes[1].axhline(5,color="black",ls="--",label="frozen calibration target")
    axes[0].set_ylabel("Mean reward per env/control-step")
    axes[1].set_ylabel("|New penalty| / |old net reward| (%)")
    for ax in axes:ax.set_xlabel("Additional PPO updates");ax.legend(fontsize=8)
    axes[1].set_ylim(bottom=0)
    fig.suptitle("Treatment reward contribution;50-update display smoothing only (full CSV retained)")
    finish(fig,"reward_contributions")

    q4=read("q4_forward_comparison.json")
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for i,(ax,cmd) in enumerate(zip(axes,COMMANDS)):
        labels=[]
        for j,group in enumerate(["control","treatment"]):
            d=q4[cmd][group]["pooled_window_risk"];labels.append(f"{group}\n{d['falls']}/{d['n']}")
            if d["rate"] is not None:
                ax.errorbar(j,d["rate"]*100,yerr=error(d["rate"],[d["wilson95_low"],d["wilson95_high"]],100),fmt="o",capsize=5,color=COLORS[group])
        ax.set_xticks([0,1],labels);ax.set_title(SPEEDS[i]+"m/s");ax.set_ylabel("Q4-forward1s risk(%)");ax.set_ylim(bottom=0)
    fig.suptitle("At-risk full-window pushes only; descriptive Wilson95% assumes independence")
    finish(fig,"q4_forward_risk")

    delay=read("delay_effects.json");fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    for j,group in enumerate(["control","treatment"]):
        for i,cmd in enumerate(COMMANDS):
            d=delay[group+":"+cmd];r=d.get("relative_risk");ci=d.get("relative_risk_bootstrap95")
            x=i+(j-.5)*.2
            if r is None:ax.text(x,1,"N/A",rotation=90,ha="center",color=COLORS[group])
            elif ci is None:ax.plot(x,r,"o",color=COLORS[group],label=group if i==0 else None)
            else:ax.errorbar(x,r,yerr=error(r,ci),fmt="o",capsize=5,color=COLORS[group],label=group if i==0 else None)
    ax.axhline(1,color="black",ls="--");ax.set_xticks(range(3),[x+"m/s" for x in SPEEDS])
    ax.set_ylabel("20ms/0ms initial-trial RR");ax.set_title("Equal-weight seed risk ratio; conditional seed-bootstrap95%")
    ax.set_ylim(bottom=0);ax.legend();finish(fig,"delay_relative_risk")
    (ROOT/"plots_complete").write_text("Six PNG/SVG scientific figures from complete data.")
    print("ISSUE12_PLOTS_COMPLETE",flush=True)

if __name__=="__main__":main()
