import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,'scripts/locomotion')
import aggregate_issue17 as analysis
old=json.loads(Path('validation_artifacts/stage1_issue15_20261111/seed_statistics.json').read_text());realroot=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
with tempfile.TemporaryDirectory(prefix='issue17_acceptance_') as tmp:
 root=Path(tmp);analysis.ROOT=root;(root/'raw_analysis_audit.json').write_text('{"passed":true}')
 raw={}
 for group in ['control_int','sym_int']:
  for protocol,commands in [('clean',['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']),('mirror',['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']),('combined',['forward_03','forward_05','forward_08'])]:
   for seed in range(1,6):
    for es in [1,2,3]:
     for cmd in commands:
      row=dict(old['control6000'][cmd][str(seed)])
      if protocol=='mirror' and row[analysis.SIGNED] is not None:row[analysis.SIGNED]=-row[analysis.SIGNED]
      raw[f'{protocol}:{group}_s{seed}:eval{es}:{cmd}']=row
 eq={'count':102400,'results':[dict(group=g,training_seed=i,mean=1.) for g in ['control6000','control_int','sym_int'] for i in range(1,6)]};(root/'equivariance_statistics.json').write_text(json.dumps(eq));(root/'job_statistics.json').write_text(json.dumps(raw));analysis.main();a=json.loads((root/'acceptance.json').read_text());assert not a['primary_pass'] and not a['gates']['action_E'] and not any(a['gates']['mirror_'+c] for c in analysis.MAIN)
 for es in [1,2,3]:raw[f'clean:sym_int_s3:eval{es}:forward_05'][analysis.ABS]=None
 (root/'job_statistics.json').write_text(json.dumps(raw));analysis.main();a=json.loads((root/'acceptance.json').read_text());p=json.loads((root/'paired_comparisons.json').read_text());assert p['primary'] is None and not a['primary_pass'] and not a['gates']['heading_forward_05']
 import plot_issue17 as plot
 import shutil
 from PIL import Image
 plot.ROOT=root
 shutil.copyfile(realroot/'old_clock_stance_distribution.json',root/'old_clock_stance_distribution.json')
 (root/'statistics_complete').write_text('ISOLATED_SYNTHETIC_FIXTURE_ONLY')
 plot.main()
 binding=json.loads((root/'figure_outputs_sha256.json').read_text());assert binding['figure_count']==6 and len(binding['outputs'])==12
 for name in binding['outputs']:
  if name.endswith('.png'):
   with Image.open(root/name) as im:im.verify()
 print('ALL6FIGURE_PIPELINE_WITH_MISSING_ENDPOINT_PASS')
(realroot/'acceptance_integration_test.json').write_text(json.dumps(dict(passed=True,isolated_fixture_only=True,no_formal_results_generated=True,strict_zero_difference_not_improvement=True,missing_training_seed_endpoint_not_dropped=True,all6figure_generation_and_PNG_decode_passed=True),indent=2)+'\n');print('ACCEPTANCE_BOUNDARIES_AND_MISSING_SEED_PASS')
