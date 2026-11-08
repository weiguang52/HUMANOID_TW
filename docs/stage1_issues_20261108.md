# Stage1 issues #1–4 experiment plan

Scope: all four GitHub issues created on 2026-11-08 around 06:54 UTC. SuperMimic remains stopped with 0/7 historical policy pilots accepted; stage1 is a separate velocity-control task.

## Evaluation protocol

Deterministic inference, seeds 1/2/3, 128 randomized environments per command, 20 seconds per trial. Commands: stand, forward 0.1/0.2/0.3 m/s, yaw +/-0.3 rad/s. Extended comparison adds 0.5/0.8 m/s and +/-0.8 rad/s. A failure is any non-timeout termination during a trial; report failed trials/128 per command and seed. Issue #1 requires each rate below 1%. Average XY norm error, yaw absolute error, root height, joint acceleration RMS and contact-conditioned foot speed are reported. Timesteps where an environment resets are excluded from continuous metrics; failures remain counted. Observation noise and startup mass/friction randomization remain enabled. Nominal testing disables external pushes; challenge testing uses identical robust08 randomization, action delay, noise and +/-0.3 m/s pushes for every compared model.

For improvement comparisons, report all seeds and commands, not only pooled means. The predeclared practical yaw target is at least 20% lower mean absolute error on the +/-0.3 yaw commands without increasing per-command fall rate by more than one percentage point; independent <1% acceptance remains a separate stricter gate. Robustness non-regression uses the same one-percentage-point fall-rate tolerance on old-range commands; failures at expanded commands are reported explicitly rather than hidden by aggregation. These are bounded exploratory experiments (two training seeds for yaw/entropy and one training seed for range/robustness), not evidence of universal robustness.

## Bounded training plan

Each formal run uses 4096 environments and 1500 PPO iterations. No checkpoint cherry-picking: evaluate the final saved checkpoint. All initial control/entropy/yaw arms resume the same FT4 model_4498 checkpoint and full optimizer, without resetting policy standard deviation. Seeds 1 and 2 are paired across arms. Checkpoint inspection gives mean action std 0.0289625 at FT4 and 0.0349594 at FT5. The historical FT5 invocation `/root/z.sh` set TW_RESET_STD=0.3, whereas the new matched arms deliberately retain the loaded FT4 std and full optimizer. Therefore direct historical FT5 versus new-run differences also include exploration initialization; causal entropy/yaw comparisons use the matched new arms. Control entropy=0; entropy arm=0.001; yaw arm additionally doubles yaw reward weight from 0.5 to 1.0. These are opt-in profiles; the default task remains unchanged.

From yaw seed1, compare range05 against robust05 at matched seed1 and 1500 iterations. Both extend forward speed to 0.5 m/s and yaw to +/-0.5 rad/s. Robust05 additionally uses +/-0.3 m/s pushes, wider startup friction/mass randomization, and per-episode uniform 0/20 ms action delay; existing observation noise is retained. Range08 and robust08 continue their respective 0.5 arms to 0.8 m/s and +/-0.8 rad/s, again matched seed1 and 1500 iterations. This estimates the combined robustness package, not separate causal effects of each randomizer. Wider-range experiments are not merged into defaults without evaluation evidence.

The baseline FT5 policy is also tested on the exact extended nominal/challenge protocols after the main queue; this provides old-policy comparisons at the new commands and left/right contact/height/acceleration symmetry diagnostics.

## Video and source references

Record colored front/side views using the same robot asset; verify original USD attributes and non-material relationships before simulation. Camera follows translation and heading; include the saved joint-color legend. Stage1 velocity commands have no HumanML3D NPY motion source, so a source skeleton comparison is not applicable. Do not imply this is an imitation acceptance test.

## Execution and limitations

Artifacts stay on `/root/gpufree-data/tw_stage1/issue_runs_20261108`. Runner stops on failed subprocesses and refuses to overwrite incomplete training runs. Initial environment smoke passed two iterations but inherited a diagnostic PYTHONPATH containing env_supermimic (RSL-RL 5.4.2); those preliminary evaluations are isolated and excluded. The launcher now clears PYTHONPATH before loading the original env_isaaclab (RSL-RL 5.0.1, torch 2.7.0+cu128), and all formal baseline/training/evaluation runs use that environment. Action-delay unit checks passed zero-delay, one-step-delay and per-environment-reset behavior. Long runs wait for the baseline evaluations and integration/video smoke tests.

The user replaced Linear synchronization with Notion on 2026-11-08. Progress and decisions are recorded at https://www.notion.so/403325d8231d8172995bf19510b8d760. AGENTS.md now requires Notion, and GitHub issue #4 records the superseding decision.

## Final execution status

All 10 formal runs completed 1500 logged iterations; checkpoint contents are finite and hashed. All planned standard/challenge evaluations, baseline symmetry/extended supplements and 32 full-duration colored front/side videos completed. See [full results](../validation_artifacts/stage1_issues_20261108/README.md) for measured improvements, failures and limits. No perturbation acceptance or default-policy promotion is claimed. SuperMimic remains stopped.
