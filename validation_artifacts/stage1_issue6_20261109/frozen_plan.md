# Issue #6 frozen experiment plan

Base commit: 7e413284c5411a72e84aa1b5acf611faa769f043.
Issue: https://github.com/weiguang52/HUMANOID_TW/issues/6 .
Only address mass randomization; defaults remain legacy.

## Intervention and audit
Old robust08 startup base_link mass: nominal + U(-0.08,0.15) kg, Isaac Lab safety clamp at 1e-6 kg. Nominal composed base mass about 0.07990783598 kg, hence the lower tail can be nonpositive.
New opt-in bounded profile: nominal * U(0.5,1.5); lower bound 50% of nominal. Same selected body, startup timing, inertia rescaling, friction static[0.4,1.4]/dynamic[0.3,1.2], delay, observation noise and rewards. This is a physically bounded research prior, not measured hardware mass identification, nor variance/mean-matched to the old additive distribution.
Audit actual initialized PhysX masses for all bodies in 4096 environments, seeds1/2, old/new. Capture pre-clamp requested masses in the real mass event only; report applied min/max/quantiles, fraction below1% nominal, exact clamp counts and pre-clamp requests. Do not infer pre-clamp draws from post-clamp floats.
Use installed Isaac Lab sampler unit tests including lower endpoint, random draws, non-target bodies, partial reset, legacy negative clamp and unchanged legacy defaults.

## Training
Same parent as #5: logs/rsl_rl/tw_stage1/2026-11-08_20-44-43_issues20261108_robust05/model_7496.pt; SHA256 c200e355d29fda60225761550ea80642bb222778ce50c00ae3deb3d95fa7a38d.
Two training seeds1/2,4096 environments,1500 updates, full optimizer resume, no std reset/checkpoint selection. Original robust08 plus stronger pushes +/-0.45m/s and bounded mass. Friction and all other components unchanged. Both seeds share one seed1 parent and are not full-pipeline independent replications.

## Evaluation
All deterministic policies; #5 paired reset seed=eval_seed*1000+command_index; eval seeds1/2/3; commands0.3/0.5/0.8m/s,128 environments,20s, original +/-0.3m/s evaluation pushes, random0/20ms delay in combined; no promotion based on training rewards.
New models: bounded combined and material-only (12 jobs).
Historical robust08 seeds1/2: bounded material-only (6 jobs), compared to original0.260%/1.563%; same checkpoints separate environment correction from retraining.
Historical strengthened-push seeds1/2: bounded combined (6 jobs), crossed against archived legacy evaluations and new bounded-trained models. Changes in trained policy and evaluated physics are not conflated.
Total24 quantitative jobs; old legacy JSONs retained as references.
Target: every eval seed at0.3 and0.5m/s strictly<1%;0.8 recorded without changing gates. Never promote defaults without both training seeds passing.
Worst instruction by summed combined falls across3eval seeds, tie order0.3/0.5/0.8. Two colored20s front/side videos per new model; camera follows position+heading, exact original physics attributes unchanged, joint legend. Velocity commands have no HumanML3D source NPY.
Artifacts stay on remote disk. Notion replaces Linear. Final measured results, failures, limitations and commit linked in issue reply.
