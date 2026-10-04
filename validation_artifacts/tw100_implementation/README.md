# TW100 phase-preserving expression pilot

Baseline snapshot 39408bd was pushed BEFORE modifying the training objective.
Original full-body task is retained. New task:
`Unitree-Custom-Humanoid-30dof-Expression-HumanML3D`.

## Implemented
- A single full-body PPO actor. Upper-body joint positions/velocities replace leg-reference commands.
- Original current L/R contact bits AND confidence are explicit observations. Same reference clock;
  no label rewriting, phase shifting, independent leg oscillator, or source time resampling.
- Contact reward remains -0.5 with existing known-mask. Soft reward is not a hard guarantee;
  low-confidence intervals are not claimed as supervised timing success.
- Upper-body position/orientation/joint rewards. Root/chest velocity and height/orientation goals.
- Remove leg/foot keypoint tracking, leg joint/segment velocity tracking and foot-reference termination.
  Height/orientation fall checks and wrist tracking termination remain.
- Standing-category drift cost (treadmill root speed does not classify walking as standing).
- Existing feet_slide -0.2, action rate -0.2, joint acceleration -2e-6; upper-only velocity error -0.2.
- Unchanged corrected URDF, PD, 50Hz control / 200Hz physics, target velocity limit scale 0.5.
  This last setting limits joint target changes, not video playback speed.

## Data and validation
All 1299 standing_upper training clips, equal clip sampling mass using weight/(frames-1).
Contact, known and source_contact4 slices copied unchanged; source hashes verified.
4 new regression tests + 9 existing contact tests passed.
Actual Isaac Lab PPO smoke: 32 environments, all standing training data, 2 iterations passed.
Initial smoke failed closed on a manifest lacking contact sidecar; corrected to training manifest.
No successful smoke policy trained on validation clips. Smoke weights are NOT a final policy.

## Stage-one comparison
Two fresh policies, seed42, same 1299 clips, 1024 environments, 6000 iterations each:
GPU0 expression objective; GPU1 original full-body objective with the same contact weight.
This compares a method change (observations + rewards), not a single scalar ablation.
New observation dimensions prohibit resuming old full-body weights.
Checkpoint every500; script refuses overwrite. Completion means trained_validation_pending.
Do not promote to walking automatically. First validate standing stability AND upper-body fidelity;
then include walking with original contact timing and explicit treadmill command semantics.
Swing clearance/impact and grouped smoothness are deferred separate ablations, not implemented here.
Final acceptance needs held-out motions, multiple seeds, original NPY + robot videos and contact metrics.
TW100 remains In Progress; no claim of improved physical performance from a two-iteration smoke.
