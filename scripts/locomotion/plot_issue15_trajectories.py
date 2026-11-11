"""Show actual world displacement/heading for the two unfiltered median-yaw trials."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from issue15_common import ROOT,sha
def main():
 assert (ROOT/"reconstruction_complete").exists()
 state=json.loads((ROOT/"reconstruction_state.json").read_text());fig,axes=plt.subplots(2,2,figsize=(14,10));inputs={}
 for row,(group,s) in enumerate(state.items()):
  with np.load(s["trajectory"]) as z:
   values=z["values"];valid=z["flags"][:,0]
  xy=values[:,:2];q=values[:,3:7];yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2));previous=np.r_[s["heading_start_yaw"],yaw[:-1]]
  increments=np.arctan2(np.sin(yaw-previous),np.cos(yaw-previous));drift=np.cumsum(np.where(valid,increments,0))*180/np.pi;t=np.arange(1,1001)*.02
  shown_xy=xy.copy();shown_xy[~valid]=np.nan
  a=axes[row,0];a.plot(shown_xy[:,0],shown_xy[:,1],color="#225ea8");a.scatter(*xy[0],marker="o",color="green",label="first post-step");a.scatter(*xy[-1],marker="x",color="red",label="20s")
  if (~valid).any():a.scatter(xy[~valid,0],xy[~valid,1],marker="x",color="gold",label="reset positions, path broken")
  for i in range(0,1000,100):
   if valid[i]:a.arrow(xy[i,0],xy[i,1],.08*np.cos(yaw[i]),.08*np.sin(yaw[i]),head_width=.025,color="#f03b20",length_includes_head=True)
  a.set_aspect("equal",adjustable="datalim");a.set_xlabel("world-relative x (m)");a.set_ylabel("world-relative y (m)");a.legend(fontsize=8);a.grid(alpha=.2)
  a.set_title(group+" forward0.3 | arrows: heading every2s")
  b=axes[row,1];b.plot(t,drift,color="#225ea8",label="reset-excluded cumulative heading")
  b.axhline(0,color="black",lw=.8);b.set_xlabel("time (s)");b.set_ylabel("heading drift (deg)");b.set_title(f"train seed{s['training_seed']} eval seed{s['eval_seed']} env{s['env_id']}")
  for i in np.flatnonzero(~valid):b.axvline(t[i],color="red",alpha=.5,lw=.6)
  b.legend(fontsize=8);b.grid(alpha=.2);inputs[s["trajectory"]]=sha(s["trajectory"])
 fig.suptitle("Nearest group-median yaw MAE across all initial trials; no survival filter\nExact recorded states; reset jumps excluded from heading integral, not silently removed from the trajectory.")
 fig.tight_layout();out=ROOT/"figures";out.mkdir(exist_ok=True);outputs=[]
 for suffix in [".png",".svg"]:
  p=out/("representative_world_heading"+suffix);fig.savefig(p,dpi=150);outputs.append(str(p.relative_to(ROOT)))
 plt.close(fig)
 (ROOT/"trajectory_figure_sha256.json").write_text(json.dumps({"sources":{str(Path(__file__)):sha(__file__)},"input_trajectories":inputs,"outputs":{p:sha(ROOT/p) for p in outputs}},indent=2)+"\n")
 print("ISSUE15_TRAJECTORY_PLOT_COMPLETE",flush=True)
if __name__=="__main__":main()
