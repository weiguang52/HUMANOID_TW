# Active reward invariance audit

Reward functional invariance under spatial reflection plus halfcycle, not a claim of exact physical dynamics conjugacy. Archived control configuration; new runtime active set/weights must match.

|Term|Weight|Invariant|Reason|
|---|---:|---|---|
|track_lin_vel_xy|3.0|Yes, under stated reflected-state assumptions|Yaw-frame planar target/velocity error reflects lateral sign; squared Euclidean norm and exponential unchanged.|
|track_ang_vel_z|1.0|Yes, under stated reflected-state assumptions|Target and axial yaw rate both negate; squared error unchanged.|
|alive|0.5|Yes, under stated reflected-state assumptions|Alive indicator unchanged under a reflected corresponding termination state.|
|base_linear_velocity|-2.0|Yes, under stated reflected-state assumptions|Vertical linear velocity unchanged and squared.|
|base_angular_velocity|-0.5|Yes, under stated reflected-state assumptions|Axial roll negates, pitch unchanged; squared planar norm unchanged.|
|flat_orientation_l2|-2.0|Yes, under stated reflected-state assumptions|Gravity lateral component negates; squared planar norm unchanged.|
|base_height|-10.0|Yes, under stated reflected-state assumptions|World vertical height and scalar target unchanged.|
|joint_acc|-2.5e-07|Yes, under stated reflected-state assumptions|Signed joint permutation preserves summed squares of accelerations.|
|action_rate|-0.05|Yes, under stated reflected-state assumptions|Same signed permutation at both consecutive actions preserves summed squared difference; actual reflected action scales equal.|
|dof_pos_limits|-5.0|Yes, under stated reflected-state assumptions|Reflected hard/soft limits swap/sign consistently; summed bound violation invariant; source limit symmetry verified in issue16 and must be asserted in runtime preflight.|
|joint_deviation_arms|-0.5|Yes, under stated reflected-state assumptions|Arm selection closed under LR swap; defaults reflect; summed absolute deviations invariant.|
|joint_deviation_waist|-1.0|Yes, under stated reflected-state assumptions|Central joint signs with symmetric defaults preserve summed absolute deviations.|
|joint_deviation_neck|-0.5|Yes, under stated reflected-state assumptions|Central joint signs with symmetric defaults preserve summed absolute deviations.|
|joint_deviation_hips|-0.5|Yes, under stated reflected-state assumptions|Selected hip/ankle yaw/roll set closed under LR swap; absolute deviation sum invariant.|
|gait|0.5|Yes, under stated reflected-state assumptions|Integer20-step phases swap exactly under+10steps; contacts swap,command norm unchanged. Strict actual-function4000state test passes.|
|swing_height|3.0|Yes, under stated reflected-state assumptions|Integer swing masks swap exactly under+10steps; world vertical foot heights swap,command norm unchanged. Strict actual-function4000state test passes.|
|feet_slide|-0.2|Yes, under stated reflected-state assumptions|Contact masks and feet permute; horizontal speed norms unchanged under reflection; sum unchanged.|
|undesired_contacts|-1.0|Yes, under stated reflected-state assumptions|Non-foot body selection closed under LR swap; force norms and threshold indicator unchanged.|

Not active in control6000 and must remain inactive in both new groups.
