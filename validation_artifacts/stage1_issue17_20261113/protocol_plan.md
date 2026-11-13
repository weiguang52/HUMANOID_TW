# Issue17 prospective protocol

Base commit: c073475561787be7b81a87f4fa7af7fc46fd86de. Scope: symmetry data augmentation only; no mirror loss, reward/network/observation/distribution changes. Training transform is spatial reflection plus half-cycle phase inversion; issue16 evaluation remains pure spatial reflection with unchanged scalar clock.

Formal training: exactly five existing requested4500 parents matched to control6000 seeds,4096 environments,1500 updates each (7500 total). Restore actor/critic/optimizer/std; never reset std or choose intermediate checkpoints. Off-switch two-update reproduction and correctness preflights are validation, never candidate selection or extra formal budget. Freeze source/config/model hashes after implementation and mandatory tests, before formal launch.

Mandatory prelaunch: actor101/critic134/action30 involution; actual observation-order and scale assertions; full reward-term invariance audit; numerical feet_gait/feet_swing_height reflection plus10control-step phase shift; off-switch vs untouched control configuration two-update bitwise reproduction.

State-set protocol: control6000 seed1..5,eval seed1,forward0.5 original clean replay;1024 environments at20 fixed post-step indices0,50,...950 per model =102400 states, with no survival or error selection. Source command ordinals/reset seeds preserved. Reproduction jobs are state-collection validations, not new acceptance denominators. Store exact raw actor states and hashes before evaluating new models. Evaluate baseline/new actors on the SAME complete state set, using deterministic means and training half-cycle transform. Report mean per-state E=norm(pi(s)-Ma*pi(Ms(s)))/max(norm(pi(s)),1e-8), median/p95 and small-denominator count; retain per-model/per-source-seed values. No rollout needed for E itself.

Formal new evaluation: five final models x3evaluation seeds x3protocols =45jobs: original clean six commands; pure-spatial mirror clean six commands; bounded combined forward0.3/0.5/0.8. Each1024environments x20s. Total225commandrecords/230400initialtrials. Preserve original command reset ordinals and paired reset rules. Reuse committed control baseline clean/mirror/combined records without alteration; verify actual config equivalence except the training symmetry switch.

Primary: forward0.5 clean mean absolute reset-aware20s heading, paired new-minus-control training-seed bootstrap95 upper <0. Five training seeds are sample units; average three evaluation seeds within each. Use same issue16 censoring: no-survivor endpoints stay NA; never drop missing seeds or call prefixes full20s. Report survivor counts/falls/exposure and seed95 upper separately.

All-pass requires: stand/0.3/0.5/0.8 mean absolute heading <=30deg; corresponding seed95 upper reported (not substituted for mean gate). Pure-spatial absolute(O+M) decreases with paired95upper<0 for each zero-yaw command; training-transform E decreases. Clean0.5/0.8 xy relative degradation <=5%; yaw+/-0.3 tracking degradation <=10%. Bounded combined0.3 fall-rate training-seed95upper <1%;0.5 fall rate no higher than control. Report paired differences/ratios and intervals, unadjusted. No default promotion unless every gate passes.

If primary fails: stop this hypothesis, no weight scan/budget addition/phase change; recommend only the issue16 no-training original/reflected-policy xoriginal/strictly symmetric-USD2x2 diagnostic, not implemented here. All other failures remain explicit even if primary passes.

Media: original new-group forward0.5 trial nearest pooled median absolute reset-aware20s heading across all15360initialtrials; no silent survivor filtering. If censored, explicitly use/label prefix-based representation and report missing full endpoint. Colored front/overhead20s source-state replay with worldXY trajectory and heading, exact source binding and manual full-resolution QA. All artifacts remain remote, with lossless Git archive, final model/optimizer checkpoints and analysis; issue reply and Notion tracking, never Linear.


# Authoritative amendment: issue17 comment6831246933

This amendment supersedes the initial unchanged-clock/five-run/control6000-primary design above. Authorization is preserved in issue_comments_updated.json; no formal training has occurred before this amendment.

Clock: feet_gait,feet_swing_height,gait_phase use one integer clock in both new groups. period_steps=round(period/step_dt)=20; idx=episode_length_buf%20; leg_idx=(idx+round(offset*20))%20; stance leg_idx<11, assert threshold*20 is integral. Observations sin/cos(2*pi*idx/20). Period0.4s,dt0.02,offsets[0,0.5],threshold0.55,weights and all other configuration remain unchanged. Preserve legacy/default behavior for unrelated tasks and old control reproduction; explicit experiment switch selects integer clock for both new groups.

Mandatory tests before formal launch: all integer steps0..999 without excluding boundaries, actual two rewards exactly equal under LRswap+10steps; each leg exactly11stance steps per20step cycle, double stance preciselyidx0and10; old floating-clock per-leg stance-count distributions across50cycles and asymmetric-cycle fraction; full active-reward invariance table. Retain prior failed floating-clock evidence as forensic preflight, never relabel it pass. Transform tests and switch-off two-update reproduction remain required.

Formal training: control_int5seeds plus sym_int5seeds, same five requested4500 parents and matched seeds,4096env1500updates each, total15000formalupdates. Both integer clock; only sym_int enables training reflection+halfcycle augmentation, mirrorlossFalse. Restore actor/critic/optimizer/std,no std reset, no checkpoint selection/parameter scan. Old control6000 is reused, zero old-group retraining.

Formal new evaluation doubles to90jobs:10finalmodels x3eval seeds x(originalclean6commands,pure-spatialmirrorclean6commands,boundedcombined3commands);450commandrecords,460800initialtrials. State-set selection stays the same102400fixed oldcontrol states, shared by oldcontrol/newcontrol/newtreatment E measurement, no state reselection.

Primary is now paired sym_int-minus-control_int forward0.5 mean absolute reset-aware20s heading; training-seed bootstrap95upper<0. Every prior relative-to-control6000 acceptance condition now uses control_int: all-four30deg mean gate withseed95upperreported; paired improvement in absolute(O+M) for zero-yaw commands and E decrease; clean0.5/0.8xy relative<=5%;turnyaw relative<=10%;boundedcombined0.3fallrate seed95upper<1%,0.5fallrate no higherthancontrol_int. The control_int-minus-oldcontrol6000 comparison is secondary descriptive only and cannot determine acceptance. Censoring/seed units/bootstrap definitions unchanged.

No formal or extra candidate updates beyond10x1500; preflight two-update reproduction is separate validation, never candidate selection. If primaryfails,stophypothesiswithoutscan/extra budget/phase scheme changes;sole futureproposal original/reflectedpolicy xoriginal/strictly symmetricUSD2x2diagnostic. Do not promote defaults unlessallgatespass. Full new models/optimizer/data/analysis/media remotely archived and pushed; issue andNotion updated, noLinear.
