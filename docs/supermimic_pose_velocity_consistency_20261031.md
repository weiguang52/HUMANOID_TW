# Pose–velocity consistency: useful direction, frozen replacement rejected

The world model independently predicts root position, root rotation, joint position and their velocities. A read-only feasibility screen uses all original trajectory and new branch transitions, without changing model weights or physics.

Three rules were fixed before validation: trapezoid(alpha0.5), endpoint(alpha1), and one scalar per group estimated on fit transitions only. The fit coefficients are translation0.684059, world rotation0.821989 and joints0.960747. Parameters were not selected or altered on validation. Actual terminal-velocity reconstructions are oracle bounds, not deployable predictions.

Using the same candidate's predicted endpoint velocities and fixed fit coefficients:

| Held data | Quantity | Independent pose | Integrated pose |
|---|---|---:|---:|
| Original trajectories,4470frames | root positionRMSE | 0.333mm | 0.0661mm |
| | root orientationp95 | 0.007074rad | 0.001923rad |
| | joint positionRMSE | 0.000673rad | 0.000297rad |
| New branches/zeros,1200frames | root positionRMSE | 0.1415mm | 0.0243mm |
| | root orientationp95 | 0.004156rad | 0.004130rad |
| | joint positionRMSE | 0.000256rad | 0.000170rad |

Every predefined rule is reported, including trapezoid regressions on new branches. Small one-step errors do not establish long-horizon consistency.

## Frozen rollout diagnostic

The explicit diagnostic wrapper reconstructs poses from predicted velocities, preserving the original velocity outputs. It is never implicitly selected by policy training or inference. Coefficients are bound to the fit-audit provenance and weights hash. No weights or checkpoint artifacts were modified.

Held branch sole-normal xy derivative cosine0.6253→0.8592 and sole-cost cosine0.1401→0.5395. Original local branch checks pass, but trajectory nonregression fails5/18 against initialH8,7/18 against the same-budget control and6/18 against the identical candidate's independent pose outputs. H32root position error increases28.4% and orientation14.8% versus that candidate. Thus frozen posthoc reconstruction is rejected as a direct replacement.

Numerical checks on all1200recorded held transitions: float64NumPy versus Torch maximum absolute error2.055e-7, exact preservation of velocity fields, unit quaternions, finite nonzero action gradients. These are implementation checks, not physical-accuracy or control acceptance.

Remote evidence root: /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics

- integration_consistency_v1: fixed rules, fit coefficients and all-source one-step errors.
- pose_integration_diagnostic_v1: complete trajectory gates, all held branch conditions, numerics and overall rejection.

Next experiment should fit the model with the same pose–velocity relationship throughout training, rather than apply it only after fitting. Keep coefficients fixed, use matched data/budget, require unchanged full gates and real control validation before adoption. No policy promotion;7motions/full durations/3independent training seeds remain unmet.
