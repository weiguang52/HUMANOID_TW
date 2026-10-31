# Offline action-branch comparison: local improvement, overall rejection

Both arms start from the same H8 offline artifact, with seed42, fresh Adam1e-4, 1000 additional updates, batch128 and final-only evaluation. Each update uses64 original fit H8 windows and64 one-step transitions, equal loss weights and576 targets. The only between-arm variable is the one-step source: original fit3305 transitions versus new fit1200 branch/zero transitions. Termination and normalizer tensors remain exactly frozen. Actors and critics are not instantiated.

Both final artifacts have54finite tensors and3249original trajectory windows. No validation data enter optimization. All source hashes and input counts are recorded. New branch data are explicitly not concatenated into episodes.

| Diagnostic | Result |
|---|---|
| Control vs initial H8, 18 trajectory checks | 3 fail |
| Candidate vs initial H8, 18 trajectory checks | 6 fail |
| Candidate vs same-budget control, 18 trajectory checks | 8 fail |
| Held branch state and derivative checks | 9/9 pass |

All1200 held branch/zero transitions and8 derivative conditions are evaluated, without selection. Relative to control, root angular-velocity derivative RMSE1.4863→1.1445, joint velocity0.21967→0.07620 and sole-cost0.0006611→0.0003767. Yet candidate sole-cost derivative cosine is only0.1401, root angular-velocity0.7391 and joint-velocity0.9976. Smaller derivative RMSE does not establish a reliable support direction.

Additional read-only decomposition reports sole-normal xy derivatives: candidate cosine0.6253, versus control0.3946. A local oracle correction using each world's recorded zero-action response barely changes sole-cost cosine0.1401→0.1423. This oracle is not deployed or counted as model prediction/control success. Error cannot be attributed solely to a constant predicted-pose bias. Both original and expanded reports are retained; original checks are unchanged.

Overall status: reviewed_failed_global_consistency. No policy training or promotion with these artifacts. Fixed7motion original-duration acceptance and3independent training seeds remain unmet.

Remote evidence root: /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/standing_branch_pair_v1

- control and candidate: fit plans, input provenance, final artifacts, integrity, all trajectory checks.
- branch_validation.json: original held-branch gate.
- local_bias_decomposition.json: extra normal-direction and oracle diagnostics.
- stage_result.json: overall rejection, distinct from the passing local branch gate.

Next: isolate inaccurate root/foot orientation response before selecting another intervention; do not treat local branch improvement as permission to train or promote a policy.
