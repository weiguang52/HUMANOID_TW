# H32-only offline calibration: rejected

Original1000 parent92a3be6e, same8fit traces and3057completeH32 windows, seed42/1000updates/batch128/freshAdam1e-4, pure rollout loss; only horizon8→32. Final-only artifact a5e8127fab7a3491ef1ff2ec59c4be839f560a705410d11a8760911985bcce32.54finite tensors; all termination-head and normalizer tensors exactly unchanged, dynamics changed. No actor/critic instantiated or policy artifact/promotion.

Prespecified gates: original source17/18pass; H8calibration nonregression6/18pass; additional H32improvement+other nonregression5/18pass. All gates fail overall. Compared with H8calibration validation episode medians:

|Metric|H8fit|H32fit|
|---|---:|---:|
|H1root position p95|1.251mm|3.812mm|
|H8root position p95|4.260mm|7.851mm|
|H32root position p95|70.262mm|48.684mm|
|H32root orientation p95|0.29428rad|0.23727rad|

H32orientation reduction19.38% misses fixed20%; do not relax. H32joint velocityRMSE increases67.3%. Long-horizon root improvement does not compensate short-term/velocity regression. Do not use for policy training.

All660existing fixed physical action branches rescored for H8offline andH32offline models, alongside the two frozen online finals. At1s small perturbations.002/.005/.01, sole-direction gradient cosine: H8offline about−0.103/−0.095/−0.102; H32offline0.063/0.071/0.075; online final controls around0.44. No reliable support-preserving action-gradient claim. Score is sum(1−world-z sole cosine), not full tracking/survival/critic objective. All branches retained; tiny initial gradients and amplitude dependence reported. Older missing contexts explicitly reconstructed event-free, not claimed recorded.

Remote evidence root: /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
- standing_wm_calibration_h32_v1: integrity, all18checks per gate, final artifact, plan/fit_input/stage_result.
- sole_sensitivity_h32_v1 and sole_sensitivity_h8offline_v1: all11gradient rows and source hashes.

Next: inspect independent action excitation in fit versus online data before another training modification. Full7motion original-duration acceptance and3independent training seeds remain unmet. No strategy promoted.
