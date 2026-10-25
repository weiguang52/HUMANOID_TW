# Upper-body drift telemetry analysis

Recorded: 2026-10-25T10:58:15.474342+08:00. Project: TW-RL. Analysis only; no training or policy promotion.

## Scope and evidence
Frozen original27995 and baseline28494; existing perturbation_20261025 telemetry. Synthetic static motion2 and upper-body motion3, with fixed leg reference; these are not HumanML3D source motion videos. Zero-disturbance seed42 provides phase comparison; 12 nonzero upper-body cases are retained in analysis.json as sensitivity data, not independent population samples.

## Findings
In the upper-body diagnostic, net XY displacement from1s to20s is35.06mm for original and102.15mm for baseline. The main baseline movement occurs5-10s:95.25mm versus26.08mm original. In this interval double-foot contact fraction is47.0% versus62.4%, and left-foot contact fraction56.1% versus84.7%. Baseline left-foot horizontal displacement is approximately121mm and its height range7.15mm. Contact threshold is force norm greater than1N.

Whole-episode contact-conditioned foot-link horizontal speed is LOWER for baseline (13.47 versus20.20mm/s); observed torque saturation fraction is zero. These measurements do not support attributing the regression simply to increased average foot speed or observed torque saturation. They do not rule out control or contact problems.

Baseline mean per-sample arm joint RMS error to policy target is0.0165rad, whereas policy target-to-reference error is0.632rad. Original values are0.0189 and0.697rad. These synthetic-reference differences alone do not establish a coordinate bug. Upper-body torso-relative position error is69.08mm versus72.50mm original, so aggregate arm tracking does not uniformly deteriorate.

## Decision and next step
Do not add a slip reward based solely on foot-link speed. First inspect sole orientation and actual contact-point tangential velocity during6-10s, distinguishing lifting, rolling and sliding. If necessary instrument a frozen-model diagnostic replay, retaining identical references and perturbation settings. Only after this distinction choose one intervention and compare against the original policy with standing, upper-body, walking and turning acceptance gates. Optimization remains in progress.

## Limits and validation
Associations are not causal evidence. No COM/support polygon/contact-point telemetry is available. Physics joint samples and control reference can differ by up to20ms. Foot-link speed is not contact-point slip. One synthetic trajectory and two checkpoints do not establish general locomotion performance. Previous baseline training was interrupted and resumed.

Verified four zero-disturbance net displacements against existing results within0.001mm; identical arm reference speed across models; four time windows per case; reviewed the timeline image. analysis.json contains all metrics and12 disturbed upper-body traces. Reproduce on remote with python scripts/practice9/analyze_upper_drift.py.
