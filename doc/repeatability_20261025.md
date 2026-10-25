# Frozen checkpoint diagnostic repeatability
User-authorized evaluation only, Notion004/TW-RL. No training, no promotion.
Models: original27995 and latest interrupted-run baseline28494. Hashes recorded
in state/audit.json. Same HeadLocked physics, baked colored USD, manifest, control.
Two synthetic diagnostics:static motion2 and upper motion3, each up to20seconds.
Each model: seeds42/123/2026 batch and individual-start cases; separate42batch
repeat, reversed-order42batch, and front-view42batch. Twelve process starts/model,
18metric/video outputs/model. All side runs enable cameras consistently.
Comparisons distinguish exact repeatability, order/startup effects, and evaluation
seed sensitivity. Different seeds include physical randomization; not training variance.
Batch-vs-single effects combine reset/order/RNG advancement: difference alone does
not prove a reset bug. Scope does not establish locomotion/general task success.
Representative front/side videos for both evaluated motions with color legend;
synthetic diagnostics are not original HumanML3D motion replays. Previous locomotion
source videos remain in previous reports, not relabeled as these references.
Acceptance: per-case termination/time/metric/drift differences, encoding checks,
visual review; compare distributions without checkpoint selection. Await results.
State:/root/gpufree-data/datasets/practice9/repeatability_20261025

## Results 2026-10-25 09:39 Asia/Shanghai
All36reports/videos complete; both frozen checkpoint hashes match. Encoding checks
passed; sampled colored front/side frames show full body and attached small parts.
All diagnostic runs complete20s without termination. Same-seed repeat differences
are exactly0; single/batch/reverse comparisons differ by at most1.12e-11 in logged
metrics and0mm net drift. No evidence of order contamination in these two tests.
IMPORTANT:standing_diagnostic.configure disables pose/velocity/joint perturbations,
observation corruption, COM/default-position randomization and fixes friction.
Thus three seeds are NOT evidence of randomized robustness; earlier expectation
of seed-varying physical randomization did not apply to this diagnostic mode.
Start-sample drift original122.69/43.06mm, baseline49.92/114.23mm (static/upper).
Prior reports discard first50control steps (1s); matching that definition gives
original109.153/35.056mm and baseline36.842/102.147mm, reproducing historical values.
Both definitions are now exposed. Static gain and upper drift regression reproduce;
not explained by diagnostic process ordering. No general locomotion or stochastic
robustness conclusion. Next evaluation should enable and audit actual perturbations,
then training-seed variance; no new training launched or formal checkpoint replaced.
