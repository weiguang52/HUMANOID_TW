"""Archive remote budget study only after evaluation, videos and all audits pass."""
import json,csv,hashlib,re,shutil,math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue10_20261109')
DEST=Path('validation_artifacts/stage1_issue10_20261109')

def metrics(old,dest):
 a=EventAccumulator(str(old),size_guidance={'scalars':0});a.Reload();out={}
 for tag in a.Tags()['scalars']:
  vv=a.Scalars(tag);assert all(math.isfinite(x.value) for x in vv),(old,tag);tail=vv[-100:]
  out[tag]={'first_step':vv[0].step,'last_step':vv[-1].step,'last':vv[-1].value,'last100_mean':sum(x.value for x in tail)/len(tail)}
 assert out;(dest/'training_metrics.json').write_text(json.dumps(out,indent=2))

def main():
 assert (ROOT/'pipeline_complete').exists() and (ROOT/'all_training_evaluation_video_complete').exists()
 assert '\nOK\n' in (ROOT/'statistics_tests.log').read_text()
 for name in ['video_physics_manual_audit.json','config_continuation_audit.json','input_checkpoint_audit.json']:assert json.loads((ROOT/name).read_text())['passed']
 models=json.loads((ROOT/'input_models.json').read_text());g=json.loads((ROOT/'gate_decision.json').read_text());a=json.loads((ROOT/'acceptance.json').read_text());stats=json.loads((ROOT/'seed_level_statistics.json').read_text())
 assert len(models)==(25 if g['gate_passed'] else 15);assert len(json.loads((ROOT/'video_quality_audit.json').read_text()))==len(models)*2
 for n,e in models.items():assert hashlib.sha256(Path(e['checkpoint']).read_bytes()).hexdigest()==e['sha256']
 for e in json.loads((ROOT/'reused_evaluation_provenance.json').read_text()).values():assert hashlib.sha256(Path(e['source']).read_bytes()).hexdigest()==e['sha256']
 DEST.mkdir(exist_ok=True)
 for p in ROOT.iterdir():
  if p.is_file() and (p.suffix in ['.json','.csv','.md','.npz'] or p.suffix=='.log' and p.name!='archive.log'):shutil.copy2(p,DEST/p.name)
 for folder in ['figures','videos']:shutil.copytree(ROOT/folder,DEST/folder,dirs_exist_ok=True)
 shutil.copy2(ROOT/'issue10_video_contact_sheet.png',DEST/'issue10_video_contact_sheet.png')
 shutil.copy2('validation_artifacts/stage1_issue9_20261109/joint_colors.json',DEST/'joint_colors.json')
 for n,e in models.items():
  d=DEST/'models'/n;d.mkdir(parents=True,exist_ok=True);shutil.copy2(e['checkpoint'],d/Path(e['checkpoint']).name)
 sources=json.loads((ROOT/'source_runs.json').read_text())
 for seed,e in sources.items():
  old=Path(e['run_dir']);d=DEST/'training_sources'/f'seed{seed}_original';shutil.copytree(old/'params',d/'params',dirs_exist_ok=True)
  for p in old.glob('events*'):shutil.copy2(p,d/p.name)
  metrics(old,d)
 if g['gate_passed']:
  for n,e in json.loads((ROOT/'continuation_state.json').read_text()).items():
   old=Path(e['run_dir']);d=DEST/'training_sources'/f'{n}_continuation';shutil.copytree(old/'params',d/'params',dirs_exist_ok=True)
   for p in old.glob('events*'):shutil.copy2(p,d/p.name)
   metrics(old,d)
 rows=list(csv.DictReader((ROOT/'training_seed_curves.csv').open()));stages=sorted(set(int(x['target_updates']) for x in rows));commands=['forward_03','forward_05','forward_08']
 assert len(rows)==len(stages)*15
 lines=['# Issue #10: requested training-budget curves and conditional continuation','','Issue: https://github.com/weiguang52/HUMANOID_TW/issues/10 . Base2e1a2ae. All artifacts generated remotely. Defaults unchanged.','','## Frozen design and provenance','','Five requested training seeds1/2/3/4/5 from issues6/9 are all retained. No applied policies, new randomization/reward changes or seed selection. All share one robust05 parent; this is conditional budget analysis, not independent parent replication.','','Nearest saved targets500/1000 are model8000/model8500: actual505/1005 completed logged updates from the initial7496 label. Final1500 is model8995. RSL checkpoints are saved after the labeled update; the label is not the number of updates in this branch. Small offsets are disclosed and chosen before results. Target updates are used on the plot, with actual counts in the table.','','Unchanged issue7 bounded-combined protocol:1024env, eval seeds1/2/3, forward0.3/0.5/0.8m/s,20s, deterministic, paired resets, requested last_action, evaluation push+/-0.3m/s. The15 final1500 eval jobs are reused byte-for-byte from hash-checked issue7/9 archives; legacy issue7 files predate last_action metadata, so requested policy/critic empty params are verified from saved configs. New evaluations use the same requested semantics. See reused_evaluation_provenance.json. Each seed/speed point has3072trials. StatisticalN=5training seeds, not trial count.','','## Gate fixed before intermediate outcomes','','D=0.5m/s mean at target1000 minus mean at1500. Threshold=half the endpoint1500 cross-seed sampleSD(ddof1). The issue did not specify which SD timepoint; endpoint1500 was prospectively frozen in the plan and Notion before reading intermediate outcomes. SD1000 and paired-changeSD are reported only as context. EnterB iff D>0 and D>threshold (1e-12rate tolerance for numerical equality).','','| Quantity | Value (percentage points unless mean) |','|---|---:|',f"| 1000 mean fall rate | {g['mean1000']:.4%} |",f"| 1500 mean fall rate | {g['mean1500']:.4%} |",f"| Decline1000-minus1500 | {g['decline']*100:+.4f} pp |",f"| SD1500 | {g['sd1500']*100:.4f} pp |",f"| HalfSD1500 threshold | {g['half_sd1500']*100:.4f} pp |",f"| SD1000 (context only) | {g['sd1000_context']*100:.4f} pp |",f"| Paired-changeSD (context only) | {g['paired_difference_sd_context']*100:.4f} pp |",f"| Gate passed | {g['gate_passed']} |",f"| Decision | {g['decision']} |",'']
 if g['gate_passed']:
  lines+=['All five own1500 checkpoints and full optimizer states were resumed for exactly3000 additional updates,4096env, no std reset. Recipe is otherwise identical. Target3000 uses model10500(actual3006logged total); target4500 uses model11994(exact4500logged total). RSL resumes the saved iteration label once, so labels repeat at the resume boundary. Five continuation config audits allow only run/log names, resume source and max_iterations, and verify optimizer step growth. No checkpoint is chosen by performance. Policy/critic/optimizer/iteration state is restored; simulator and rollout state are reconstructed under the same seed at the resume boundary, so this is not a byte-identical uninterrupted4500-update trajectory.','']
 else:lines+=['The frozen gate failed. **No continuation training was launched; stepB was skipped.** "Converged" here means the issue-defined operational stopping rule, not proof of asymptotic convergence or no possible benefit from longer training.','']
 lines+=['## Per-training-seed curves','','Rates pool exactly3072trials per seed/speed/checkpoint. Within-policy Wilson intervals in all_evaluation_rates.csv are descriptive evaluation uncertainty, not training-seed uncertainty.','','| Target updates | Actual logged | Speed | Seed | Falls /3072 | Fall rate |','|---:|---:|---|---:|---:|---:|']
 for x in rows:lines.append(f"| {x['target_updates']} | {x['actual_logged_updates']} | {x['command']} | {x['training_seed']} | {x['falls']} | {float(x['fall_rate']):.4%} |")
 lines+=['','![Five-seed curves with group mean and sampleSD](figures/budget_curves.png)','','## Group means, sampleSD and seed-level intervals','','| Target updates | Speed | Mean | SampleSD (pp,ddof1) | Seed-bootstrap95% mean |','|---:|---|---:|---:|---|']
 for stage in stages:
  for c in commands:
   s=stats[str(stage)][c];lo,hi=s['bootstrap95'];lines.append(f"| {stage} | {c} | {s['mean']:.4%} | {s['sample_sd']*100:.4f} | [{lo:.4%},{hi:.4%}] |")
 if g['gate_passed']:
  d=a['mid_speed_decline1500_minus4500'];lo,hi=d['bootstrap95'];lines+=['','## StepB acceptance',f"- 4500-update0.3m/s mean95% upper strictly below1%: {a['low_speed_mean95_upper_below1pct']}.",f"- 0.5m/s decline1500-minus4500: {d['mean']*100:+.4f} pp, paired seed-bootstrap95% [{lo*100:+.4f},{hi*100:+.4f}] pp; excludes0: {d['excludes_zero']}."]
 if g['gate_passed']:lines+=['','Middle-speed improvement is supported by the prespecified paired interval, but low-speed acceptance fails. The4500 low-speed mean is worse than1500; longer training does not improve every speed monotonically. The3000 low-speed upper happens below1%, but that checkpoint is not selected or promoted after observing the curve. The final4500 result and every seed remain in the report.','','![Prespecified late checkpoints, no selection](figures/budget_late_curves.png)']
 lines+=['','## Statistical limits','','All5^5=3125with-replacement seed-index vectors are enumerated. The same indices resample both checkpoints for change intervals, preserving seed pairing. Linear2.5/97.5percentiles form intervals.3125bootstrap draws do not add training seeds. With only five observed seeds, tail/coverage uncertainty is large. SeedSD includes finite evaluation error; fixed evaluation streams and a shared parent limit generalization. Secondary speed/timepoint intervals are not simultaneous familywise95% intervals. The stopping gate is heuristic and was not converted into a hypothesis test. No defaults are promoted.','','## Colored worst-command videos','','For every evaluated checkpoint, the command with the highest sum of falls over evaluation seeds1/2/3 gets front/side diagnostic videos; ties use0.3,0.5,0.8 order. Existing final1500 videos are hash-validated remote copies from their original archives and are explicitly identified in video_plan.json. New videos use one environment/20s; they are visual diagnostics, not1024-env acceptance results. Cameras follow translation and heading. Exact precoloredUSD preserves original physics, limits and non-material relationships; no dynamic material binding. Colors:joint_colors.json. Stage1 velocity commands have no HumanML3D sourceNPY (N/A).','']
 videos=sorted((DEST/'videos').glob('*.mp4'));assert len(videos)==len(models)*2
 for p in videos:lines.append(f'- [{p.stem}](videos/{p.name})')
 lines+=['','## Validation and tracking','','Five targeted tests cover gate direction/strict equality/endpointSD, joint seed resampling, sampleSD and invalid data. Input checkpoint SHA/iteration/finite tensors/nonempty optimizer state, baseline source identity, saved configuration and continuation semantics are audited. Video encoding/frame/color/framing and manual attachment review accompany the report. The input/model hashes and archive manifest allow independent verification. Original raw logs remain remote; archive logs remove ANSI and trailing whitespace. No mounted data was copied to local storage.','','Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . GitHub issue reply records the gate, whether training actually ran and the final evidence.']
 (DEST/'README.md').write_text('\n'.join(lines)+'\n')
 for p in DEST.rglob('*.log'):
  s=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',p.read_text(errors='replace'));p.write_text('\n'.join(x.rstrip() for x in s.splitlines()).rstrip()+'\n')
 manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='sha256.json'};(DEST/'sha256.json').write_text(json.dumps(manifest,indent=2));print('ARCHIVED',len(manifest),'files;GATE',g['decision'],'MODELS',len(models),'VIDEOS',len(videos))
if __name__=='__main__':main()
