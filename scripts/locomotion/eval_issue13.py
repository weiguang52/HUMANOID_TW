"""Analysis-only in-memory observer adapter; original evaluator/configs remain unchanged."""
import difflib,hashlib,subprocess,sys
from pathlib import Path
BASE="0ec49f358d1634b62b7f8b149381809b54d23b9b"
original=Path(__file__).with_name("eval_cmd.py")
source=original.read_text()
assert original.read_bytes()==subprocess.check_output(["git","show",BASE+":scripts/locomotion/eval_cmd.py"])
source_sha=hashlib.sha256(original.read_bytes()).hexdigest()
replacements=[
 ('args_cli, hydra_args = parser.parse_known_args()',"""parser.add_argument("--issue13_records",default="")
parser.add_argument("--issue13_capture_env",type=int,default=-1)
parser.add_argument("--issue13_capture_command",default="forward_08")
args_cli, hydra_args = parser.parse_known_args()"""),
 ('    results = []',"""    tracking = None
    if args_cli.issue13_records:
        from issue13_tracking import TrackingRecorder
        tracking = TrackingRecorder(base,runner,args_cli.issue13_records,resume_path,args_cli.issue5_condition,args_cli.issue13_capture_env,args_cli.issue13_capture_command)
    results = []"""),
 ('        writer = None',"""        if tracking is not None:
            tracking.begin(label,command)
        writer = None"""),
 ('            count += valid.sum()',"""            if tracking is not None:
                tracking.sample(values,valid,robot.data.root_lin_vel_b,robot.data.root_ang_vel_b[:,2])
            count += valid.sum()"""),
 ('        print("EVAL",json.dumps(results[-1]),flush=True)',"""        if tracking is not None:
            tracking.finish(results[-1])
        print("EVAL",json.dumps(results[-1]),flush=True)"""),
 ('"checkpoint":resume_path,"deterministic_policy":True,',""""checkpoint":resume_path,"deterministic_policy":True,
        "issue13_tracking":None if tracking is None else tracking.job_metadata(),
        "issue13_source":{"original_sha256":_ISSUE13_ORIGINAL_SHA,"instrumented_sha256":_ISSUE13_PATCHED_SHA},""")
]
for old,new in replacements:
 assert source.count(old)==1,old
 source=source.replace(old,new)
patched_sha=hashlib.sha256(source.encode()).hexdigest()
root=Path("/root/gpufree-data/tw_stage1/issue13_20261110")
(root/"adapter_source_diff.patch").write_text("".join(difflib.unified_diff(original.read_text().splitlines(True),source.splitlines(True),fromfile="original_eval_cmd.py",tofile="read_only_issue13_adapter.py")))
sys.argv[0]=str(original)
exec(compile(source,str(original),"exec"),{"__name__":"__main__","__file__":str(original),"_ISSUE13_ORIGINAL_SHA":source_sha,"_ISSUE13_PATCHED_SHA":patched_sha})
