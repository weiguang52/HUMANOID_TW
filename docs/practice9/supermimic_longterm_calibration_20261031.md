# TW SuperMimic long-term calibration: ongoing

The new goal is stable control, not just a runnable method transfer. Keep the
original seven test motions and termination conditions, require their full
requested durations and at least three independent training seeds, compare
controlled baselines, and deliver all videos. Do not promote automatically.

All artifacts below are on the data disk under
`/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics`.

## Reproducible development sequence

1. `prepare_supermimic_calibration.py`: frozen source-family split of the332
   training clips/331 families;66 families for validation. No overlap with the
   existing final holdout. Both partitions were seen by the old policy; this is
   a development split for new calibration, not fresh generalization evidence.
   Nominal inverse PD offsets across153960frames show shoulder-roll absolute
   p95 around3.69/3.60. Values outside[-1,1] are not automatically errors;
   clipping there would remove valid reference targets.
2. `collect_supermimic_calibration.py`: first four manifest entries per partition,
   selected before simulation, three controllers(nominal, hold-initial,
   reference PD targets), original terminations, no reset after failure.
   `physics_collection_v1` has24 episodes/1232 transitions; all finite and
   adjacent state/next-state continuity checked. Average durations are
   .9925/.8675/1.22s, all0/8 completed. These are diagnostic controls, not skills.
   `physics_collection_mapping_v2` additionally verifies every processed PD
   target and records action/state joint order, scale, offset and clip.
   Orders match the original checkpoint. This does not prove balance or full
   dynamic equivalence.
3. `audit_supermimic_calibration_model.py`: frozen-model H1/H8/H32 plus constant
   root-velocity baseline on each real episode, without padding/reset crossing.
   Original checkpoint baseline is `calibration_model_baseline.json`.
4. `calibrate_supermimic_world_model.py`: `wm_calibration_v1`, fixed seed42,
   1000 updates, batch128, window8, fresh Adam lr1e-4 on dynamics net only.
   Only fit trajectories are used. Normalizers and termination head are checked
   bitwise unchanged; no actor or critic is trained. Final artifact is explicitly
   offline dynamics calibration, not a resumable policy checkpoint. Only final
   update1000 is considered; no checkpoint selection.
5. `compare_supermimic_calibration.py`: predefined validation episode-median
   root-position/orientation p95 must improve>=20% at H1/H8/H32, with other
   reported RMSE medians worsening no more than5%. This is a development
   diagnostic condition, not a replacement for stable-control acceptance.

## Actual result and reporting correction

Calibration exited0 and all18 development checks passed. H32 validation median
of episode root-position p95:.29215→.16518m; orientation:2.15416→.71257rad.
Root linear/angular velocity RMSE and joint position/velocity RMSE also improve.
Absolute root error remains substantial; short and narrowly sampled episodes
cannot establish robust learned dynamics or reliable control.

`wm_calibration_v1/comparison.json` is an invalid initial comparator result:
GPU float mean of an all-true mask sometimes equals.9999999404, and comparison
with exactly1 falsely flagged nonfinite rollouts. Raw reports had no failure
and included all accuracy metrics. The comparator now recovers integer counts;
new evaluations directly record finite_count and integer-count/window fraction.
`comparison_countfix.json` is the valid result. No actual nonfinite state is
accepted and no performance gate was relaxed. Preserve both reports.

## Next work

Collect sustained nominal-pose and controlled small-perturbation trajectories
with separate seeds, plus wider motion/control coverage. Verify model quality
on new physical sampling before resuming actor optimization. A static nominal
PD calibration is not learned standing success. Existing final holdout has
already been observed; retain its reuse limitation and add a genuinely untouched
source-isolated confirmation set for final claims. Main long-term goal remains
active; existing production policies and original Isaac runtime are unchanged.


## 2026-10-31 08:02 CST: static coverage and combined calibration

Static perturbation collection produced 16 episodes / 7775 transitions. Six reached 20 seconds: four identical zero-input nominal-PD cases and two amplitude-0.02 cases. These are not learned-policy successes. All higher-amplitude cases terminated under unchanged gates.

The first motion-only calibration failed the static H8 root-position improvement gate (3.7% improvement versus the required 20%). Combined v2 therefore adds static fit trajectories, with exactly 64 motion and 64 static windows per batch. It starts from the same original model and retains 1000 updates, seed42, H8, batch128, Adam1e-4, and frozen normalizers/termination head. Validation and final holdout are not fitting data.

Combined v2 passes both 18-condition comparisons against the original model and all 18 static nonregression conditions against v1. It fails five motion nonregression conditions against v1: H1 orientation p95 +18.65%, angular velocity RMSE +5.09%, joint-position RMSE +5.25%; H8 angular velocity RMSE +5.16%, joint-position RMSE +10.09%. The prespecified 5% cap remains unchanged. Overall result: rejected for promotion.

H32 median episode root-position p95 improves from v1 0.16518 to 0.12775 m on motion validation and 0.12775 to 0.07765 m on static validation. Absolute errors remain substantial. These are offline prediction results, not closed-loop policy success. Evidence is under longterm_diagnostics/wm_combined_v2, including four *_vs_*.json files and the evaluation_protocol.json saved before evaluation.

Next: diagnose short-horizon versus rollout-loss tradeoffs and sampling coverage before another fixed-protocol experiment. Stable learned standing, stepping, all seven fixed motion durations, and at least three independent training seeds remain outstanding. Long-term goal remains active.


### Fixed teacher-forcing experiment v3

The combined collection has 522 motion and 3249 static fit windows, sampled exactly 64/64 per batch. V2 optimizes an H8 free rollout, which does not directly constrain one-step predictions from the real states at steps 2–8. This motivates a hypothesis test, not a proven explanation of its regressions.

V3 changes only the loss to 0.5 times the original H8 rollout loss plus 0.5 times the mean real-state one-step loss over the same eight transitions. All original initialization, seed, update count, optimizer, frozen parameters and group sampling remain fixed. Contiguity of every trace is asserted before using previous next_state as the teacher input. The runner records its protocol before fitting and evaluates only the final 1000-update artifact. It requires the original baseline gates and <=5% nonregression versus both v1 and v2 separately on motion and static validation. Run: scripts/practice9/run_supermimic_teacher_calibration.sh. Output: longterm_diagnostics/wm_combined_tf_v3. This produces an offline dynamics artifact, never a deployable policy checkpoint.


### 2026-10-31 08:08 CST: v3 rejected; real-physics standing baseline

V3 completed training and six comparisons with process exit0, but failed acceptance. Both original-model comparisons pass, static versus v1 passes, and the other three comparisons fail. Motion H32 position episode-p95 median is 0.17265m (+35.14% versus v2); static H32 position is +53.08% versus v2. Thus the equal-weight one-step loss does not resolve the short/long horizon tradeoff. All artifacts remain preserved; no policy loads this rejected calibration.

Started `tw_standing_v1` with `run_tw_supermimic_standing_v1.sh`: fresh seed42,256 environments,1000 iterations, unchanged online SuperMimic/physics/randomization/reward parameters. The single changed input is the synthetic1001-frame nominal-standing motion instead of332 motion clips. This tests whether the learner can acquire the simplest real-physics control task. It does not use any prior actor or offline calibration artifact. The runner automatically evaluates final checkpoint for20s against zero-offset nominal PD, recording front/side video and original termination reasons. Actual finite states and speed caps must be checked, plus drift/error reported. Success here is a synthetic training-task diagnostic only; all seven original independent motions and three independent training seeds remain required for the long-term goal. State/protocol/logs live in `supermimic_tw_20261031/standing_v1_seed42` on the data disk.

## Standing and contact follow-up, 2026-10-31

Standing v1 completed1000 iterations. Final learned policy terminates at16 steps/0.32s (ee_body_pos), nominal zero-offset PD completes1000 steps/20s. All four H264/50fps videos have exact step counts and pass full ffmpeg decode. Actual joint speeds obey caps in both trajectories; imagined H1/H8 speeds violate caps about28% of entries. Policy max absolute action2.731; root XY displacement46.7mm at0.32s versus nominalPD2.06mm at20s (different durations, not comparable20s drift). See standing_v1_seed42/evaluation/results.json, video_integrity.json and action_speed_audit.json.

Contact probe v2 records actual Warp solver contact data through get_data_into/mj_contactForce. V1 omitted robot/ prefixes and its empty contact arrays are invalid, retained for provenance. V2 has16 fixed perturbation episodes,780 samples, finite/contiguous traces,6 full20s episodes. Four zero-input duplicates are not independent learned successes. Nominal seed42 samples2-19.8s: four loaded points per foot,49x22.85mm footprint, total normal force8.248N matching robot weight. Perturbations lose support area while soles tilt25-35degrees. This does not establish a collision-model defect. URDF box extents match original STL bounds. Probe code is independent of policy training.

Source comparison must use corresponding HumanML3D ankles7/8, not toes10/11 against robot foot origins. Across332 clips, normalized-duration height correlations have medians0.987/0.992 before contact correction and0.935/0.933 afterward, none below0.5. This supports retained lift timing but is not clearance/dynamic-feasibility acceptance. Reports under longterm_diagnostics include all_source_ankle_stage_screen.json and near_ground_sole_screen.json; no training/test data were modified.

Optional imagined_velocity_penalty defaults0 and subtracts mean squared relative joint-speed excess from imagined reward only. It leaves predictions and real physics untouched. Math/gradient tests pass;8env/4iteration smoke completes with coefficient1 and191 finite checkpoint tensors. Fresh standing_velocity_v2_seed42 uses256env,1000iterations,seed42 and all v1 settings, with this coefficient as the sole experimental change. Final1000 only, automatic same20s trained/zero videos, unchanged termination. Run script and data-disk protocol are preserved. No claim of improved control until actual evaluation. Original fixed7 motions and3 independent training seeds remain outstanding.
