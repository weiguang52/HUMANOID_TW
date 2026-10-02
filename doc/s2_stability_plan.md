# S2 stability and simple-action plan

Linear: TW-65 (parent), TW-66 diagnostics, TW-67 dynamics/reference audit,
TW-68 controlled training, TW-69 multi-seed acceptance.

Keep the S1 final checkpoint as baseline. Measure four walking clips with seeds
42--46 at 50 Hz control and 200 Hz physics using run_s2_baseline.sh. Exclude
reset boundaries from acceleration and spectral metrics. The >=8 Hz power
fraction is a diagnostic, not a hardware stability certificate. Contact foot
horizontal speed is a slip proxy, including rolling/rotation of the foot.
Actuator-reported torques are simulation estimates.

Initial seed-42 clip 000801 comparison: reference joint-position high-frequency
fraction 0.00003822; target fraction 0.051528 with observation noise and 0.045985
without. Both complete the reference once without abnormal terminations.
This single comparison does not establish general improvement. Noise is not
the only source of high-frequency commands. Continue with one-factor changes
to action-rate regularization, filtering and PD, checking tracking and contact.

URDF mass is 0.840743 kg. User confirmed this is close to the real robot;
do not rescale masses/inertias. Motors are not selected. Current actuator
parameters are simulation assumptions; record required torque/speed envelopes
and revisit after motor selection. All 33 inertial links passed positivity
and principal-inertia triangle checks.

No nonwalking diagnostic clip is admitted to training: all 210 candidates
failed existing quality gates. Repair joint mapping/reference feasibility,
then introduce standing, weight shifts and waving gradually. Do not relabel
failed clips or loosen physical hard limits to pass them.

Acceptance: five seeds per action, no walking completion regression, initially
target >=30% high-frequency reduction and >=90% simple-action completion;
report tracking error, contact/slip and force peaks alongside those targets.
Only compare identical clip/seed/noise settings. These targets are provisional.

Artifacts: per version, publish the final checkpoint, validation videos,
configs, metrics and hashes. S1 v1 includes only four walking videos and
model_9999.pt; intermediate checkpoints and nonwalking diagnostic videos
remain on the server. Do not copy mounted data to the local computer.

Validation: 80 existing practice9 tests and 8 subtests pass. Spectral helper
checks separate 2 Hz and 12 Hz signals and handle constant/invalid windows.
100-step smoke produced 400 physics samples; full multi-seed runs are in tmux.
