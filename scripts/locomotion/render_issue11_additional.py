"""Complete added full-joint representatives after the original queued replays."""
import json,time,os
from run_issue11 import ROOT,REPO,run,evaluate
os.chdir(REPO)
while not (ROOT/'analysis_and_replay_complete').exists():time.sleep(20)
models=json.loads((ROOT/'input_models.json').read_text())
for mode in [3,4]:
 selection=ROOT/f'representative_mode{mode}.json';s=json.loads(selection.read_text())
 for view in ['front','side']:
  output=ROOT/f'mode{mode}_{view}.json'
  if output.exists():continue
  args=evaluate(models[s['model']]['checkpoint'],s['eval_seed'],output,1)+['--video','--camera_view',view,'--trajectory_replay',str(selection)];args[args.index('--command_filter')+1]='forward_03';run(args,output.with_suffix('.log'))
(ROOT/'all_representative_replays_complete').write_text('complete');print('ALL_FOUR_CASE_REPLAYS_COMPLETE',flush=True)
