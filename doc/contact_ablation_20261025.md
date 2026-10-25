# Stationary contact-loss ablation

TW-RL / Notion004, authorized continued optimization2026-10-25.
Evidence: contact-point diagnostic matches previous zero-disturbance root trajectories
exactly. During5-10s baseline upper motion left positive-load coverage62%, point
horizontal RMS speed mean0.034mm/s. Foot origin moves123.27mm forward in unloaded
or transition samples versus-2.54mm in loaded-to-loaded samples. This supports
investigating support dropout, not increasing a velocity-based slip penalty.
Contact loss and movement association is not proof of why the policy unloads the foot.

One opt-in reward term: weight-0.25, bounded mean squared shortage below1N over both
feet and sensor history. Eligible only for audited standing references with both
contact labels known/on. Continuous reference XY speed gate1 below.005m/s,0 above
.025; yaw gate1 below.1rad/s,0 above.3. Walking, turning and unknown contact labels
are excluded. No observation/action, friction, timing, reset or physical changes.
Existing contact phase reward stays unchanged. Nine pure gate/history checks pass.

Matched baseline HeadLocked and candidate StandingContact start original27995,
seed42,1024env,precommitted500iterations. Both restore checkpoint optimizer learning
rate scalar, removing the known mismatch consistently. Sampler/RNG resume remains
incomplete; single seed limits inference. Do not compare against an older baseline
as if it were matched. No fixed-test checkpoint selection or early favorable stopping.
Two64env/2iteration smoke runs required before formal execution. Formal final checkpoint
then full42case colored side/front evaluation and source references using verified
prior template; preserve any shortened failure videos.

Acceptance: static20s drift <109.15mm and new matched baseline; upper20s complete
and drift <=original35.06mm and matched baseline; standing6/6,walking>=23/24,turning3/3.
Inspect upper/wrist tracking for regression, not merely total success or reward.
No automatic production promotion. If rejected, retain original and inspect evidence.
State /root/gpufree-data/datasets/practice9/contact_ablation_20261025.

## Intermediate result,2026-10-25 12:54+08
All42 metric evaluations completed. Matched baseline:standing6/6,flat24/24,
turning3/3,reference-limit diagnostic3/3,treadmill0/3,nonplanar0/3.
Candidate matches other counts but turning1/3: unequivocal rejection; no promotion.
Candidate000211 seed42 first termination5.74s(anchor_pos+ee_body_pos), seed123
5.40s(anchor_ori); matched baseline all finish12.12s. Orientation termination
is not synonymous with yaw failure; seed123 yaw alone does not explain it.
Reference category walking means the new reward's standing gate is false throughout.
This excludes direct evaluation-time gate activation, not training interference.
State turning_failure_audit.json and metric_counts_prevideo.json retain evidence.
Drift diagnostics and videos still pending; do not infer a support benefit yet.

Potential next mechanism, not implemented: preserve original action behavior on
locomotion while learning balance. Primary research reference: Proximal Policy
Distillation https://arxiv.org/abs/2407.15134 combines student-driven distillation
and PPO, but does not establish success on this robot. ProtoMotions fine-tuning
supports frozen pretrained modules (not a drop-in guarantee of behavior retention):
https://github.com/NVlabs/ProtoMotions/tree/main/protomotions/agents/fine_tuning.
Any implementation needs an explicit matched comparison and unchanged test gates.
