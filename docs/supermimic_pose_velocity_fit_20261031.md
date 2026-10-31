# Pose–velocity structure fitted throughout offline calibration: rejected

Same originalH8 parent,64 original fitH8 windows+64 new fitH1 transitions,1000 updates,576 targets/update, seed42, samplers42/43, freshAdam1e-4 and fixed fit-derived integration coefficients. The only change against standing_branch_pair_v1/candidate is applying pose reconstruction during both H8 and H1 optimization, rather than using independently predicted poses. The previous control endpoint is not loaded for continuation.

The artifact has a distinct kind offline_pose_velocity_calibration_not_policy_checkpoint and explicit prediction_contract=pose_from_velocity_fit_alpha_v1. Raw velocity-network weights alone are not this model: explicit audited loaders must apply the saved coefficients and world-rotation integration. Ordinary policy continuation still rejects the incompatible artifact kind. Nothing automatically promotes or trains a policy.

All fit source hashes match the4505fit transitions used to estimate the coefficients. Termination and normalizers remain exactly frozen; final artifact finite checks pass. No validation fitting or checkpoint selection.

Results:

- InitialH8 trajectory nonregression fails5/18.
- Same-budget independent-pose control nonregression fails6/18, including H32root position ratio1.271 and orientation1.163.
- Held branch checks fail1/9: root angular-velocity derivative reduction misses the fixed20% threshold.
- Held root angular-velocity derivative cosine0.6853, joint velocity0.9963, sole-normal xy0.8585 and sole cost0.5748. These local improvements do not establish globally consistent dynamics or stable control.

The wrapper refactor was checked against float64NumPy on all1200recorded held states: max difference2.055e-7, unchanged velocity fields, unit quaternions and finite nonzero action gradients. Three malformed/mismatched contracts are rejected; the ordinary-model path returns the original model unchanged. These are implementation checks only.

Remote evidence: /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/pose_velocity_fit_v1

State reviewed_failed_integrated_fit, passed=false, policy_training_started=false, promoted=false. Preserve all artifacts. Next inspect which root-response errors remain and their fit-loss contribution before selecting another intervention. Do not tune integration coefficients on validation. Fixed7motions/full original durations/3independent training seeds remain unmet.
