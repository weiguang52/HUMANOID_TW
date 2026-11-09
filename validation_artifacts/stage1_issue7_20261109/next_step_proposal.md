# Proposed next single modification (NOT implemented)

Change only the policy observation `last_action` to the actual last applied action term:
`ObsTerm(func=mdp.last_action, params={"action_name":"JointPositionAction"})`.
Leave critic observations, observation dimension/network architecture, rewards, push/mass/friction/noise/delay distributions and evaluation protocol unchanged.

The current no-argument Isaac Lab `last_action` returns `env.action_manager.action` (last requested input). Supplying the term name returns that term's `raw_actions`. In the existing delay action class, `super().process_actions(delayed)` stores the delay-selected command there. Thus the proposed single parameter exposes which normalized target was actually applied in the previous control step, without a privileged future-delay flag or extra observation dimensions.

Evidence:83.115% of failures are within1s after a push with sagittal-dominant terminal tilt;20ms-delay trials fail1877/18260 (10.279%) versus948/18604 (5.096%) at0ms, with the same direction in all four policies. Existing near-limit flags are present in essentially all at-risk states and do not distinguish failures. These observations motivate testing actuator-feedback timing, not a claim that the observation change will fix the failures.

Future test only: compare unchanged observation vs this one observation parameter, fixed shared robust05 starting point and equal training budgets/seeds, then rerun the identical1024-env challenge. Verify0ms observation equivalence and20ms applied-action timing first. Real controller replay would need to supply the same applied-target history. No part of this proposal is implemented or trained in issue7.
