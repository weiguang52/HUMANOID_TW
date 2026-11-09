"""Translate measured risks without conflating trial coverage and push coverage."""
import csv,json
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue11_20261110')
rows=list(csv.DictReader((ROOT/'all_push_records.csv').open()));low=[x for x in rows if x['stage']=='4500' and x['command']=='forward_03' and x['eligible']=='True'];q=[x for x in low if x['amplitude_q']=='4'];qf=[x for x in q if x['direction']=='forward'];total=15360;nf=158
coverage={'scope':'descriptive unit-translation of existing marginal factors; not a new acceptance or independently validated corner','ampQ4_exposed_trials':len({x['trial_id'] for x in q}),'ampQ4_proximal_failures':sum(int(x['failure_within1s']) for x in q),'ampQ4_forward_exposed_trials':len({x['trial_id'] for x in qf}),'ampQ4_forward_proximal_failures':sum(int(x['failure_within1s']) for x in qf),'all_low_trials':total,'all_low_first_failures':nf}
for key in ['ampQ4','ampQ4_forward']:coverage[key+'_trial_fraction']=coverage[key+'_exposed_trials']/total
(ROOT/'amplitude_trial_coverage_diagnostic.json').write_text(json.dumps(coverage,indent=2))
corners=json.loads((ROOT/'corner_diagnostics.json').read_text())['4500:forward_03'];e=corners['envelope'];remaining=e['remaining_trial_risk'];cv=json.loads((ROOT/'secondary_concentration_cv.json').read_text());effects=list(csv.DictReader((ROOT/'factor_effects.csv').open()));delay=next(x for x in effects if x['unit']=='20s_trial' and x['stage']=='4500' and x['command']=='forward_03' and x['factor']=='delay_ms' and x['target_level']=='20')
text=f"""Low-speed failures are **not dominated by the predefined mass/friction extreme corners**. The larger envelope touches {100*e['trial_coverage']:.3f}% of initial trials but precedes only {e['proximal_failures']}/{nf} ({100*e['proximal_failure_fraction']:.3f}%) first failures. Excluding WHOLE exposed trials leaves {remaining['falls']}/{remaining['n']}={100*remaining['rate']:.4f}% (descriptive Wilson95% [{100*remaining['wilson95_low']:.4f}%,{100*remaining['wilson95_high']:.4f}%]); it is not a new acceptance test.

This does **not** mean uniform risk. Large pushes in Q4 touch {100*coverage['ampQ4_trial_fraction']:.3f}% of initial trials, so a quarter of pushes is not a tiny trial subset. The observed Q4+forward subset touches {100*coverage['ampQ4_forward_trial_fraction']:.3f}% of trials and precedes {coverage['ampQ4_forward_proximal_failures']}/{nf} ({100*coverage['ampQ4_forward_proximal_failures']/nf:.3f}%) low-speed first failures. This is substantial dynamic-stress enrichment, not proof of an infeasible physical-parameter corner or independent confirmation of a discovered subset. The held-out-seed diagnostics capture {100*cv['baseline']['held_out_failure_capture']:.3f}% of failures with {100*cv['baseline']['held_out_trial_coverage']:.3f}% baseline coverage, and {100*cv['push']['held_out_failure_capture']:.3f}% with {100*cv['push']['held_out_trial_coverage']:.3f}% push-subset trial coverage. Neither meets the frozen >=80%capture/<=20%coverage criterion; 'broader' means residual failures across measured physical settings, not absence of predictive pushing states.

At4500,20ms delay is associated with a seed-mean trial risk ratio {float(delay['seed_mean_relative_risk']):.3f} versus0ms (conditional seed-bootstrap95% [{float(delay['rr_seed_boot95_low']):.3f},{float(delay['rr_seed_boot95_high']):.3f}]). The amplitude-Q4 window has133/9068 events, whereas Q1 has0/9116; an infinite risk ratio is not manufactured from the zero reference. Single support occupies36052/36404 eligible windows; its large CASE count alone is not evidence of a large support-phase risk effect. Double-support references and some phase bins are sparse, with wide or non-estimable intervals.

Longer training reduced observed no-recent-push low-speed cases26->2 and20ms side/right-support cases20->3, while20ms forward/left-support cases grew15->51 and forward/right-support19->39. These are same-label sample changes, not proof that a mechanism was eliminated. At4500,140/158 low-speed failures are both within1s of a push and terminally sagittal. All158 failing initial mass/friction/delay settings have a successful other-seed trial, but exact pushing/COM/phase trajectories differ: **exact failed-state physical recoverability remains unidentified**.

The single next proposal is a **low-speed forward-push recovery reward targeting pitch/COM restoration**, tested under unchanged acceptance, rather than assuming a few mass/friction combinations are impossible. This issue does not implement the proposal, train models or promote defaults.
"""
(ROOT/'interpretation.md').write_text(text)
reps=[]
for i in range(1,5):
 p=ROOT/f'representative_mode{i}.json';s=json.loads(p.read_text())
 if s['mode']=='no_recent_push':ee=[];n=None;k=None
 elif s['selection_scope']=='coarse_case_mode':
  ee=[x for x in low if f"delay{x['delay_ms']}__{x['direction']}__support{x['support_code']}"==s['mode']];n=len(ee);k=sum(int(x['failure_within1s']) for x in ee)
 else:
  ee=[x for x in low if all(str(x[f])==str(v) for f,v in s['joint_stratum'].items())];n=len(ee);k=sum(int(x['failure_within1s']) for x in ee)
 s['eligible_push_events']=k;assert n==s['eligible_push_denominator'];s['case_count_is_not_the_push_window_numerator']=s['selection_scope']=='coarse_case_mode';p.write_text(json.dumps(s,indent=2));reps.append(s)
(ROOT/'representative_selection_summary.json').write_text(json.dumps(reps,indent=2));print('INTERPRETATION_COMPLETE',flush=True)
