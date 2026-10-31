# Action excitation and fixed orthogonal branch collection

Fit-only state→action ridge screen uses lambda0.001 and fit normalization, with all30action dimensions and all original16files retained. Fit3305frames: medianR2=0.999701, residualRMS/fitActionStd=0.017285. Validation4470frames: medianR2=0.747757, residualRatio=0.522276. Nonzero-forcing subsets retain1305/2470frames, R2=0.999704/0.747703 and residualRatio=0.027384/0.702591. These are linear association statistics, not proof of nonlinear non-identifiability or a unique failure cause. Long zero segments are reported separately rather than silently dominating the conclusion.

## New physical data

orthogonal_branches_v1 contains960fit branches,960validation branches and480zero responses, all40batches verified. Same synthetic static reference and original physics/velocity projection/terminations,60worlds, no random events. Fit zero-PD warmups5/25/75/150steps; validation15/35/125/175. Every30action axes receives±0.02/±0.05 for one physical step, then signs swap across the paired worlds. Every warmup also has a full zero-action baseline. No failures or sample selection.

Maximum same-world repeated-start error6.659e-6; cross-world paired-start difference5.847e-6, both under the existing1e-5 numerical replay bound. After subtracting per-world zero responses, swapped-assignment derivative cosines for root velocities and joint positions/velocities all exceed0.99999996. All original state/action/context/next_state arrays, labels, model mapping and source hashes retained.

Contract is branch_transitions_not_episodes_v1. Do not concatenate as trajectories or feed the original H8/H32 window loader. Validation is a static temporal split with convergent/nearby states, not independent-motion/hardware generalization. The old660diagnostic branches and fixed7motion tests are not added to fit. New data have not been used for model or policy training.

## Numerical follow-up

Sole score sum(1−cosTilt) is now computed equivalently as2*sum(qx²+qy²), avoiding float32 cancellation near zero. All660old diagnostic branches rescored; prior reports retained. H8offline small-amplitude warmup1s cosine remains about−0.103/−0.101/−0.104; H32offline0.072/0.068/0.074. The weak sensitivity conclusion remains, with stable-form v2 reports authoritative. This correction changes neither the physics target nor any acceptance threshold. It is not the complete tracking/survival/critic policy gradient.

Evidence root: /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
- action_excitation_v1/audit.json: all dimensions/files and spectra.
- orthogonal_branches_v1/integrity.json: complete fixed grid, source hashes and swapped-assignment checks.
- sole_sensitivity_h8offline_v2 and sole_sensitivity_h32_v2: stable-form complete11row comparisons.

Next: design a matched-budget offline continuation control and fit-only branch-mixture calibration, keeping validation and old probes separate. Verify dynamics/gradient consistency before further policy training. Original7motions/full durations/3independent training seeds remain unmet; no policy promoted.
