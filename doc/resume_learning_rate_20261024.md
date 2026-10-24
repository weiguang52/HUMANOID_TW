# Resume scheduler audit and matched pilot

Notion004 / TW-RL. Initiated2026-10-24 Asia/Shanghai.
Problem observed in installed RSL-RL PPO.load: optimizer state restored, scalar
self.learning_rate not restored. PPO.update uses that scalar during adaptive KL
adjustment and writes it back to optimizer groups. The original model_27995.pt
optimizer lr is2.25e-5, while BasePPORunnerCfg initializes scalar lr to1e-3,
a44.44x discrepancy before the first adaptive update. Actual first-step ratio
can differ because KL immediately adjusts it. This is a demonstrated state
restoration mismatch, not proof it explains all evaluation regressions.

Opt-in --restore_optimizer_learning_rate synchronizes scalar from restored
optimizer AFTER runner.load and records before/after in resume_learning_rate.json.
No model weights, optimizer moments, reward, physics, sampling or schedule changes.
Reject missing/nonpositive/nonfinite or heterogeneous optimizer group rates.
Default remains unchanged for reproducible matched baseline; intended only for PPO.

Tests:27 selected tests pass, including8 resume tests, wrap/gate heading tests and
support-velocity tests. Real simulator smoke is required for each variant before
formal training. Both start from same original checkpoint, not smoke outputs.
Pilot:baseline/resume500iterations,seed42,1024env,save every100; no new reward.
Final checkpoint predetermined. Both run42metric cases and46colored/source videos.
Acceptance:20second upper diagnostic, static drift below109.15mm and baseline,
standing6/6, flat at least23/24, turning3/3 retained. No automatic promotion.

Other recovery limitations: checkpoint keys are actor,critic,optimizer,iter,infos;
sampler failure/visit counters and environment RNG/physics state are not restored
by this path. Their effect is not isolated here and is not silently changed.
One-seed short continuation is evidence for this run, not a robust general claim.

State:/root/gpufree-data/datasets/practice9/resume_ablation_20261024
Scripts:train_resume_ablation.sh,prepare_resume_evaluation.py,run_resume_experiment.sh.
Status:smoke running; formal result pending.
