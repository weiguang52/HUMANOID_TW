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


## Fit-loss and actual-contact follow-up

loss_balance_v2 decomposes the exact existing state loss on all3305original fit transitions,1200fit branch/zero transitions and3249complete original H8 windows. Group contributions sum to the actual loss within numerical tolerance; frozen normalizer scales are recorded. In the actual0.5H8+0.5H1 objective, root angular velocity contributes19.99% for the independent model and34.09% for the integrated model. The latter quaternion term contributes17.57%, joint velocity25.77%. This does not support a simple omitted/vanishing root-loss explanation; loss fractions are not gradient contributions or causal proof. No weights changed.

contact_response_errors_v2 aligns all780native GPU contact samples to their own post-step transitions, retaining16traces, source hashes and original forcing split. No CPU contact recomputation. Threshold1e-4N defines loaded points;3plus is a point-count label, not a declaration of full face contact. Validation has389both-feet3plus samples,59some-feet2orless and no airborne samples.

Integrated root angular-velocity one-step RMSE is0.02426rad/s versus0.11999rad/s in the two observed support groups. But actual state changes and forcing amplitudes differ, so raw error growth is not contact causality. Within all7validation amplitude/support strata, the learned prediction is worse than a constant-angular-velocity baseline. For amplitude0.02, baseline versus model is0.00411/0.02710rad/s with3plus points and0.00567/0.03103with fewer points; amplitude0.05 fewer-point0.05185/0.09668; amplitude0.1 fewer-point0.08235/0.24353. Nominal zero forcing is0.00134/0.01668. Full per-frame and per-forcing results retained. The baseline assesses prediction only, not a control skill.

Next isolate solver-history effects before assuming a missing geometric feature or changing loss weights. Source review shows velocity projection followed by storing qacc_warmstart, while the learned state does not include this solver cache. This observation alone is not a physics bug: a dedicated one-step probe is required before any physics change. All original physics and policies remain untouched.
