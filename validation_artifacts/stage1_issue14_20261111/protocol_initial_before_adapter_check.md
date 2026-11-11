# Issue #14: critic-only recovery-clock matched continuation

Source: https://github.com/weiguang52/HUMANOID_TW/issues/14 (created 2026-11-11 00:36:48 UTC).
Base: 5a1f8b9a96a4b5d44cd8de2bc0900332d88e07bb; branch supermimic/tw-locomotion-20261108.

## Scope and immutable intervention
Only opt-in privileged critic timing changes. Append two columns to the critic: active recovery flag and normalized remaining time. Actor/policy remains101 dimensions, critic134 ->136; hidden layers unchanged. Original reward, its weight34.67877532210905, all original observation terms/noise, physical and delay/push distributions remain unchanged.

The new term calls the SAME push_window and push_clock as the existing reward. At dt0.02, the shared function is active for age0..50 inclusive. Remaining time is clamp(1-age/50,0,1), masked to zero outside the active window. The endpoint at age50 has active=1 and remaining=0, exactly preserving the existing reward endpoint. At age51 both are zero. Reset clears the shared timer to its existing sentinel.

Isaac step order is reward -> reset -> interval push -> next observation. The observation represents the actual post-event state: a push marked at stepk first appears in that observation at age0; its first affected reward is computed at k+1. This is NOT replaced by the #13 post-event analysis mask (ages0..49). Reward-computation timing, sentinel and original inclusive endpoint are not changed to make clocks appear aligned.

## Checkpoint and RNG migration
Construct the original134-wide critic and original rollout buffers, strictly load the ORIGINAL4500 parent checkpoint with actor, critic, full Adam state, std and iteration; audit exact equality BEFORE expansion.
Append exactly two zero columns to critic.mlp.0.weight and to its Adam exp_avg/exp_avg_sq; preserve every old element, optimizer step counter and parameter-group setting. Expand critic rollout observation storage. All other parameters and hidden widths remain identical.
No extra random initialization; actor same-state output is bitwise checked before/after expansion. No std reset or normalization change.

## Preflight
Ten unit tests: no-push zeros, EVERY boundary step vs reward, selected/all reset, repeated pushes, no RNG consumption, default-off no-op, only critic configuration change, original reward-clock prerequisite, single-delegate evaluation push, full state/Adam migration and learnable new columns.
Fixed64-env smoke jobs: exact archived legacy source2 PPO updates; new switch-off2 updates; clock-on2 updates; clock-on1000-step deterministic no-update rollout vs archived #12 treatment smoke. Six smoke updates total at64envs, excluded from formal budget; no smoke checkpoint is a formal parent/candidate. If an operational preflight fails, document/fix before formal work; no adaptive scientific search.

## Frozen formal budget and historical reuse
Use exactly the SAME five requested4500 parent checkpoints recorded in issue12/input_models.json, training seed1..5. Restore all state;4096envs;1500 additional PPO updates EACH. One final model per seed; no intermediate selection. Total7500 formal updates, no extra seeds, no weight/clock encoding search or adaptive continuation.
Historical control/treatment final models are the exact archived #12 ten checkpoints. Historical evaluation is exactly #13 clean and combined full records. Compare all training behavior/dependency sources against #12 frozen manifest (only train.py opt-in guard additions and new clock/migration support are allowed); per-seed runtime config must differ from #12 treatment only by critic clock and log/run metadata.

## Evaluation
New group only: clean five commands in original order (forward0.3,yaw+0.3,yaw-0.3,forward0.5,forward0.8); combined forward0.3/0.5/0.8. For EACH five final models x3evaluation seeds:1024envs x20seconds, deterministic actor, original paired reset seeds eval_seed*1000+command_index. Thirty new formal evaluation jobs,120commands,122880initial trials (76800clean+46080combined).
Clean actual nominal masses/friction, no pushes/noise,0ms delay must be checked. Combined retains bounded physics,0/20ms delay, original noise/pushes. Evaluation clock delegates the exact original push sampler ONCE without extra sampling.
Reuse the read-only #13 moment recorder and independent reader: full-horizon nonterminal samples incl ordinary episodes after resets; initial-episode sensitivity; post-event analysis window[push,push+1s),ages0..49 and separate reward-aligned ages1..50. Neither changes the actual training clock.
Report falls, xy/yaw, mean vx and per-trial temporal population SD(ddof0) at every speed/partition. Raw counts/sums/sumsq/masks/reference samples and all original per-trial failure records retained.

## Frozen inference and decisions
Inference unit FIVE matched training seeds, first average three evaluation-seed summaries within each seed; equal weight to five seeds. Enumerate all5^5=3125 paired joint bootstrap index samples; percentile95%, no multiple-comparison correction.
Primary: clean0.8 xy new-clock minus old-treatment paired absolute difference95% UPPER endpoint STRICTLY<0 is required to support mitigation.
Full success requires ALL:
- Primary passes.
- clean0.5 and0.8 xy new vs old-control point-mean relative degradation<=5%.
- clean0.8 mean-vx new minus control95% interval contains0 or is positive (upper>=0; no entirely negative interval).
- combined0.3 new fall seed-level95% upper STRICTLY<1%.
- combined0.5 new mean fall rate<=old-control mean.
Failure to support primary stops this hypothesis after requested evaluation/delivery. Do not adjust timing/encoding, scan weights, append updates or silently relax gates. No default promotion based on partial success.

## Secondary actual-speed/fall analysis
For each of THREE groups and each combined commanded speed, place measured mean body vx beside first-fall count/rate and paired seed-level contrasts. Include clean achieved-speed reference too. Explicitly describe achieved-speed differences and common support; do not call unequal achieved speeds matched or infer recovery causally from fall differences. Prespecify descriptive actual-speed bins using per-initial-trial forward mean (0.05m/s bins over0..1, with under/overflow), common bins and three group counts/falls; no gate selection from bins. Conditioning on achieved speed is post-treatment and survival/occupancy biased, so secondary bins do not identify a causal speed-adjusted effect. All raw per-trial moments/failure labels permit audit.

## Representative media and delivery
Each group: the combined0.8 trial nearest its group median full-horizon trial mean xy among all15360trials, no survivor filter, ties trainingseed/evaluationseed/env id. Old two selections/media exactly reused from #13; new group selected by same fixed rule. Capture the chosen original1024-env evaluation and verify formal arrays/moments exactly; do not count reconstruction as new formal trials.
Three groups x front/side colored20s exact-state replay, actual push timestamps, same source frames within group, no cross-group synchronization claim. Camera follows translation and heading, precolored exact experiment asset preserves original1876attributes; joint-color legend. Stage1 velocity commands have no HumanML sourceNPY: explicitly N/A. All products remain remote; no mounted files copied locally.
Check encoding/frame counts, sampled fullbody/colors/attachments, full-resolution response frames/contact sheets, all plots, raw/source hashes, archive/index/commit objects and actual GitHub ref/tree. Commit/push remotely, reply issue14 and append Notion results/status Done; no Linear. Preserve unrelated files and previous issue12 rejection/history.
