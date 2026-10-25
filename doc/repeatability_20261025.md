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
