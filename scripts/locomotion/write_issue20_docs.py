"""Update final status from verified results while preserving history."""
import json
from issue20_common import ROOT,REPO

def main():
 assert json.loads((ROOT/'prepublication_requirement_audit.json').read_text())['passed']
 c=json.loads((ROOT/'scientific_conclusion.json').read_text())
 records=json.loads((ROOT/'evaluation_records.json').read_text())
 sym=json.loads((ROOT/'symmetric_analysis.json').read_text())
 selected=json.loads((ROOT/'representative_selection.json').read_text())
 n=len(records);outcome=c['classification'];stop=(ROOT/'prerequisite_stop.json').exists()
 base='validation_artifacts/stage1_issue20_20261114'
 lines=['# Issue20: start phase0/10 x original/strict symmetric USD','']
 lines+=['Five exact issue19 sym_wrap final policies; zero training, no reward/default changes.','']
 lines+=[f'Completed {n} formal jobs, {4*n} command records, {4096*n} initial trials. Outcome: **{outcome}**.','']
 if stop:
  lines+=['The symmetric-USD mirrored-trajectory prerequisite failed. Stop as requested; original-USD30 jobs skipped. No P/Phi/Gamma inference.','']
 else:lines+=[f'Full phase/USD factorial complete. P={c["physical_fraction_P"]}, as defined in the issue.','']
 lines+=[c['recommendation'],'','## Protocol and validation','']
 lines+=['Each condition:1024 environments x3 evaluation seeds x20s. Commands: stand,0.3,0.5,0.8m/s. k10 uses mirrored initial states and (episode_step+10)%20 gait observations. Clean protocol has no observation corruption or extra RNG draws.','']
 lines+=['All five policies passed102400-state phase equivariance, including actual float32 clock error below1e-5. Full-size four-command1000-step k0 action/state reproduction passed; every executed original-USDk0 formal job also verifies all archived issue19 yaw arrays.','']
 lines+=['All12left bodies,12left joint local attributes and12left collision vertex arrays remain unchanged. Full physical model differences and actual PhysX readback are retained.','']
 lines+=['Statistics use five training seeds, three evaluation seeds averaged within each, and all3125 paired bootstrap resamples.95% intervals are individual, without simultaneous-coverage claims. Reset jumps are excluded; full20s and common-prefix exposure are separate.','']
 lines+=['The symmetric prerequisite requires abs(mean S)<=5deg for all commands and sign-reversed half-cycle swap residual intervals containing0: k10left=-k0right and k10right=-k0left. Numerical clock/state/tensor/solver effects remain distinct from causal physical explanations.','']
 lines+=['|Command|Symmetric USD mean S deg|95% interval|','|---|---:|---|']
 for command in ['stand','forward_03','forward_05','forward_08']:
  v=sym['paired_group_statistics'][command]['S20_degrees']
  lines.append(f'|{command}|{v["mean"]}|{v["ci95"]}|')
 lines+=['','## Results and media','']
 lines+=[f'- [Full analysis](../{base}/results.md), [seed tables](../{base}/seed_tables.md), [model differences](../{base}/usd_physical_comparison.md).']
 lines+=[f'- [Paired overhead k0/k10](../{base}/videos/phase0_phase10_overhead_paired.mp4).']
 for phase in [0,10]:
  lines+=[f'- [k{phase} front](../{base}/videos/phase{phase}_front.mp4), [k{phase} side](../{base}/videos/phase{phase}_side.mp4).']
 lines+=[f'- [Joint colors](../{base}/joint_colors.json), [lossless restoration](../{base}/RESTORE.md), [existing input objects](../{base}/existing_reference_bindings.json).']
 lines+=['',f'Video USD={selected["usd"]}, training seed={selected["training_seed"]}, chosen by maximum absolute seed-mean D. Both phases use fixed evaluation seed1/env0; no maximum-trial selection.','']
 lines+=['Exact saved poses only; display velocities zero, no dynamics integration. Common worldXY scale. Front/side follow translation and heading; overhead follows translation with fixed world orientation. Precolored model preserves the corresponding physical asset. HumanML3D NPY: N/A, stage1 velocity commands.','']
 lines+=['Four plots, two contact sheets, full-resolution arm/turn frames and paired overhead are actually reviewed. Raw data, videos, assets, intervals and hashes remain remote; large payloads are lossless chunks, while MP4 paths stay directly accessible.','']
 lines+=['Historical issue17/19 acceptance failures remain preserved. SuperMimic stays stopped. Task records use Notion; Linear is not used.']
 (REPO/'docs/stage1_issue20_20261114.md').write_text('\n'.join(lines)+'\n')
 readme=REPO/'README.md';text=readme.read_text()
 old=[x for x in text.splitlines() if x.startswith('| Isaac Lab stage1 velocity walking |')];assert len(old)==1
 state=f'Issue20 complete: {n} formal evaluations, zero training; frozen outcome {outcome}. '
 state+=('Required symmetric-USD prerequisite stop; original USD skipped. ' if stop else 'Full phase/USD factorial completed. ')
 state+='No default promotion; issue17/19 acceptance remains FAILED.'
 evidence='[Issue20 results](docs/stage1_issue20_20261114.md), [archive](validation_artifacts/stage1_issue20_20261114/results.md)'
 text=text.replace(old[0],'| Isaac Lab stage1 velocity walking | '+state+' | '+evidence+' |')
 marker='<!-- ISSUE20_STATUS_START -->';end='<!-- ISSUE20_STATUS_END -->'
 summary=marker+'\n'+state+' The same five final symmetric policies are tested with start phases0/10 and original/strict symmetric USD. '
 summary+='Rewards/defaults unchanged. [Full report](docs/stage1_issue20_20261114.md) includes all cell/seed intervals, paired S/D, startup and late slope, half cycles, model differences and colored paired-overhead/front/side videos. Historical findings below remain preserved.\n'+end
 if marker in text:
  a=text.index(marker);b=text.index(end,a)+len(end);text=text[:a]+summary+text[b:]
 else:
  anchor='\nIssue #19:';assert anchor in text;text=text.replace(anchor,'\n'+summary+'\n'+anchor,1)
 readme.write_text(text);print('ISSUE20_DOCS_UPDATED',outcome,n,flush=True)
if __name__=='__main__':main()
