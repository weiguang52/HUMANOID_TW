# Issue #8: applied-action policy observation, equal-budget two-seed comparison

Issue: https://github.com/weiguang52/HUMANOID_TW/issues/8 . Base9b6149d. Production defaults remain requested-action observation.

## Single intervention and validation

Opt-in --last_action_observation applied adds only action_name=JointPositionAction to policy.last_action parameters. Isaac Lab then returns the action term raw_actions, which the existing delay class has selected from current/previous requested action. Critic last_action still returns environment-wide requested action. No extra observation dimension or network change; no reward/distribution changes. Saved configuration audits allow only this policy parameter and run/log names relative to matching issue6 models. Checkpoint tensor shapes match issue6 exactly.

Four real64-env ActionManager tests passed:0ms requested/applied equality;20ms observation equals the delay-selected action and actual target buffer; full/partial resets clear episode history; critic/config/dimensions unchanged.64-env2-update resume smoke passed.

## Fixed training and evaluation

Two seeds1/2,4096 environments,1500updates each, full optimizer resume from the same robust05 parent SHA256 c200e355d29fda60225761550ea80642bb222778ce50c00ae3deb3d95fa7a38d; no std reset or checkpoint selection. Training pushes+/-0.45m/s and bounded mass nominal*U(0.5,1.5), original friction/noise/random0/20ms delay. Both seeds share the same seed1 parent; not full-pipeline independent replications.

Evaluation follows issue7:1024 environments,3eval seeds,0.3/0.5/0.8m/s,20s, deterministic policy, paired reset seed=eval_seed*1000+command_index. Combined evaluation pushes remain+/-0.3m/s. New6jobs/18command estimates/18432trials. Historical controls reuse issue7 same-protocol fixed-policy evaluations, not128-env estimates. No fresh requested-action control training is duplicated: matching issue6 runs already use the exact parent/budget/seeds and audited unchanged configuration. Issue6 is the primary single-change control; issue5 has a different training mass distribution and is contextual reference only.

Wilson95% intervals are conditional binomial intervals for each fixed policy/task. Pooled3072-trial intervals do not measure training-seed uncertainty, guarantee cross-policy independence, or test paired differences. Larger batches are not exact historical128-env episode replays.

## Side-by-side pooled rates and95% intervals

Every cell has denominator3072 (three eval seeds); counts, rates and Wilson interval.

| Training seed | Command | Issue5 strengthened, requested | Issue6 bounded-trained, requested | Issue8 applied observation |
|---|---|---|---|---|
| 1 | forward_03 | 26/3072; 0.846% [0.578%, 1.237%] | 41/3072; 1.335% [0.985%, 1.805%] | 66/3072; 2.148% [1.692%, 2.724%] |
| 1 | forward_05 | 104/3072; 3.385% [2.802%, 4.085%] | 121/3072; 3.939% [3.307%, 4.686%] | 269/3072; 8.757% [7.808%, 9.808%] |
| 1 | forward_08 | 470/3072; 15.299% [14.070%, 16.616%] | 473/3072; 15.397% [14.164%, 16.717%] | 602/3072; 19.596% [18.231%, 21.038%] |
| 2 | forward_03 | 21/3072; 0.684% [0.448%, 1.043%] | 25/3072; 0.814% [0.552%, 1.199%] | 43/3072; 1.400% [1.041%, 1.880%] |
| 2 | forward_05 | 144/3072; 4.688% [3.995%, 5.493%] | 230/3072; 7.487% [6.608%, 8.472%] | 64/3072; 2.083% [1.635%, 2.651%] |
| 2 | forward_08 | 520/3072; 16.927% [15.643%, 18.294%] | 650/3072; 21.159% [19.751%, 22.639%] | 398/3072; 12.956% [11.814%, 14.190%] |

## Per-evaluation-seed new-model rates

| Model | Eval seed | Command | Falls/1024 | Rate | Wilson95% interval |
|---|---:|---|---:|---:|---|
| applied_s1 | 1 | forward_03 | 21 | 2.051% | [1.345%, 3.115%] |
| applied_s1 | 1 | forward_05 | 82 | 8.008% | [6.498%, 9.831%] |
| applied_s1 | 1 | forward_08 | 187 | 18.262% | [16.015%, 20.745%] |
| applied_s1 | 2 | forward_03 | 21 | 2.051% | [1.345%, 3.115%] |
| applied_s1 | 2 | forward_05 | 92 | 8.984% | [7.383%, 10.893%] |
| applied_s1 | 2 | forward_08 | 184 | 17.969% | [15.738%, 20.439%] |
| applied_s1 | 3 | forward_03 | 24 | 2.344% | [1.580%, 3.464%] |
| applied_s1 | 3 | forward_05 | 95 | 9.277% | [7.649%, 11.210%] |
| applied_s1 | 3 | forward_08 | 231 | 22.559% | [20.104%, 25.218%] |
| applied_s2 | 1 | forward_03 | 9 | 0.879% | [0.463%, 1.662%] |
| applied_s2 | 1 | forward_05 | 18 | 1.758% | [1.115%, 2.761%] |
| applied_s2 | 1 | forward_08 | 121 | 11.816% | [9.981%, 13.938%] |
| applied_s2 | 2 | forward_03 | 17 | 1.660% | [1.039%, 2.643%] |
| applied_s2 | 2 | forward_05 | 25 | 2.441% | [1.659%, 3.579%] |
| applied_s2 | 2 | forward_08 | 119 | 11.621% | [9.800%, 13.729%] |
| applied_s2 | 3 | forward_03 | 17 | 1.660% | [1.039%, 2.643%] |
| applied_s2 | 3 | forward_05 | 21 | 2.051% | [1.345%, 3.115%] |
| applied_s2 | 3 | forward_08 | 158 | 15.430% | [13.347%, 17.771%] |

## Delay0/20ms split and frozen gap criterion

Each group pools the same balanced3speeds/3eval seeds within one training policy. The archived baseline gap was frozen before training. The gap criterion is a strict reduction of the absolute point-estimate20ms-minus0ms rate difference, not a significance claim; signed gaps and both rates expose shrinking by worsening0ms. Detailed stratified counts/CIs are in delay_trial_rates.csv.

| Model | Delay ms | Falls/trials | Rate | Wilson95% interval |
|---|---:|---:|---:|---|
| issue6_s1 | 0 | 180/4651 | 3.870% | [3.353%, 4.464%] |
| issue6_s1 | 20 | 455/4565 | 9.967% | [9.132%, 10.870%] |
| applied_s1 | 0 | 169/4651 | 3.634% | [3.133%, 4.211%] |
| applied_s1 | 20 | 768/4565 | 16.824% | [15.767%, 17.937%] |
| issue6_s2 | 0 | 384/4651 | 8.256% | [7.499%, 9.082%] |
| issue6_s2 | 20 | 521/4565 | 11.413% | [10.523%, 12.368%] |
| applied_s2 | 0 | 110/4651 | 2.365% | [1.966%, 2.843%] |
| applied_s2 | 20 | 395/4565 | 8.653% | [7.872%, 9.504%] |

## Frozen acceptance and recommendation

Per training seed: both pooled0.3/0.5m/s Wilson95% upper bounds strictly<1%, and absolute delay gap strictly smaller than matching issue6 baseline. Both training seeds must pass.

- applied_s1: upper-bound criterion FAIL;0.3upper=2.724%,0.5upper=9.808%; delay gap DID NOT SHRINK (6.097pp -> 13.190pp); signed new gap=13.190pp; all criteria FAIL.

- applied_s2: upper-bound criterion FAIL;0.3upper=1.880%,0.5upper=2.651%; delay gap DID NOT SHRINK (3.157pp -> 6.288pp); signed new gap=6.288pp; all criteria FAIL.

Both training seeds: FAIL. Defaults promoted: NO.

Frozen-rule recommendation: **stop treating this variant as a demonstrated fix; change the hypothesis rather than extending this budget ad hoc**. This is a recommendation only; no further training is launched. Evidence is the acceptance values above, including low/mid-speed non-worsening against matching issue6 control. The experiment budget is complete even when performance acceptance fails.

## First-failure trajectories and recent-push sagittal share

The unchanged issue7 pre-reset recorder/mode definitions are reused. Raw traces contain actual push delta-v/timing, delay, contacts/cycle phase, root/joints and soft/hard limit flags. First true termination per environment is counted once; at most1s available history, with early failures explicitly marked. The mode means latest push within1s plus abs(pitch)>abs(roll) at termination; it is descriptive, not a unique causal mechanism.

| Seed | Policy | Recent-push sagittal cases / all first failures | Share |
|---|---|---:|
| 1 | issue5 requested | 492/600 | 82.000% |
| 1 | issue6 requested | 452/635 | 71.181% |
| 1 | issue8 applied | 805/937 | 85.912% |
| 2 | issue5 requested | 628/685 | 91.679% |
| 2 | issue6 requested | 776/905 | 85.746% |
| 2 | issue8 applied | 408/505 | 80.792% |

Raw audit: 1442first-failure windows; 1430full1s; minimum41frames for early failures. Terminal pre-reset heights, push bounds, delay labels, finite states and checkpoint hashes validated. See failure_events.csv, failure_mode_summary.json, mode_by_policy_speed.csv and figures/ for all grouped diagnostics/evidence. Case shares are not causal relative risks.

## Colored worst-command front/side videos

For each new model, choose highest summed challenge fall count over three eval seeds, ties0.3/0.5/0.8. Videos are20s one-environment seed1 diagnostics using the applied observation; they are not1024-env acceptance. Camera follows translation and heading. Exact precolored USD retains original physics attributes/non-material relationships; no runtime binding. Velocity commands have no HumanML3D sourceNPY. Joint legend: joint_colors.json.
- [applied_s1_worst_front_forward_08_front](videos/applied_s1_worst_front_forward_08_front.mp4)
- [applied_s1_worst_side_forward_08_side](videos/applied_s1_worst_side_forward_08_side.mp4)
- [applied_s2_worst_front_forward_08_front](videos/applied_s2_worst_front_forward_08_front.mp4)
- [applied_s2_worst_side_forward_08_side](videos/applied_s2_worst_side_forward_08_side.mp4)

Video encoding/frame/framing/color checks and manual small-part review are recorded in the audit files.

## Tracking and artifact integrity

All artifacts were generated remotely; raw originals remain on the data disk. Archived logs strip ANSI decoration/trailing whitespace. No mounted files are copied to local storage. Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . Issue reply states measured gates and whether to continue/stop/change the hypothesis.
