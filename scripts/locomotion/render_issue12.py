"""Twenty colored worst-command diagnostic clips; formal evaluation remains separate."""
import json,os
from pathlib import Path
from issue12_common import ROOT,REPO,run,evaluate,sha

def main():
    os.chdir(REPO);assert (ROOT/"analysis_complete").exists()
    models=json.loads((ROOT/"final_models.json").read_text())
    selection=json.loads((ROOT/"worst_command_selection.json").read_text());plan=[]
    for name,m in models.items():
        assert sha(m["checkpoint"])==m["sha256"]
        worst=selection[name]["command"];record={"model":name,"command":worst,"checkpoint_sha256":m["sha256"],"views":{}}
        for view in ["front","side"]:
            out=ROOT/f"{name}_worst_{view}.json"
            if not out.exists():
                args=evaluate(m["checkpoint"],1,out,trace=False,num_envs=1)
                args[args.index("scripts/locomotion/eval_cmd.py")]="scripts/locomotion/render_eval_issue12.py"
                args[args.index("--command_filter")+1]=worst
                args+=["--video","--camera_view",view]
                run(args,out.with_suffix(".log"),"video_"+name+"_"+view)
            d=json.loads(out.read_text())
            assert d["checkpoint"]==m["checkpoint"] and d["deterministic_policy"]
            assert len(d["results"])==1 and d["results"][0]["command"]==worst
            assert d["results"][0]["num_envs"]==1 and d["results"][0]["seconds"]==20
            video=ROOT/"videos"/f"{name}_worst_{view}_{worst}_{view}.mp4"
            assert video.exists()
            assert "Colored USD integrity: 1876 original attributes unchanged" in out.with_suffix(".log").read_text()
            record["views"][view]={"output":str(video),"sha256":sha(video),"eval_metadata":str(out),
                 "scope":"illustrative independent one-env seed1, single command/reset index0; no formal risk estimate",
                 "camera_follows_translation_and_heading":True}
        plan.append(record);(ROOT/"video_plan.json").write_text(json.dumps(plan,indent=2))
    assert len(plan)==10
    (ROOT/"videos_complete").write_text("20 colored illustrative worst-command front/side clips.")
    print("ISSUE12_VIDEOS_COMPLETE",len(plan)*2,flush=True)
if __name__=="__main__":main()
