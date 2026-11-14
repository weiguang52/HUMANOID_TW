# Issue21: first closed-loop mirror deviation diagnosis

Status: prescribed evaluations complete. Classification: **O**.

Same five final sym_wrap policies and exact issue20 strict symmetric USD. Evaluation only: zero training updates; no reward, production configuration or default USD changes; no parameter scan. Fixed-root suspension and yaw perturbations are diagnostic conditions only.

Repair the specifically identified non-equivariant pipeline component, then rerun only the symmetric-USD issue20 prerequisite in a future issue.
The recommendation is not implemented in this issue.

## Scope and units

Completed 25/25 jobs. Full scope: five training seeds; stand and forward0.5m/s; k0/k10;1000 control steps at20ms;16 environments and independent num_envs=1 repetitions. Open-loop200 steps in both environment counts, fixed suspended base/no contact and normal ground, using saved closed-loop k0 actions and exact mirrored actions. Perturbations: signed epsilon0.01/0.05/0.2rad/s,1024 environments, evaluation seed1,20s, five training seeds.

All quantities retain full signed components. Thresholds1e-6/1e-4/1e-2 use strict component-wise absolute crossings; group L2 and Linf are also retained. Observation/action index0 is before the first control step; physical index1 is after the first control step. Observation residuals use the actual policy features after configured preprocessing/scales; they are not unscaled physical velocities. Physical positions are in meters, joint/heading angles radians (reported heading slopes deg/s), angular rates rad/s, forces N, joint torques Nm. The main applied_torque trace is PhysX projected active joint force (get_dof_projected_joint_forces), not an independently exposed pure implicit-drive torque. estimated_pd_torque preserves Isaac Lab approximate PD estimates, explicit_actuation_force preserves explicit input (which can be zero for implicit drives), and joint_target preserves the position command. These observables are kept distinct. Quaternion components are dimensionless and compared after equivalent-sign alignment. Origins are removed in float32 capture; any tiny origin-subtraction residual is reported separately.

Every command/phase/condition runs in an independent fresh process, without previous trajectory contact/solver history. Exact issue20 stored k0 initial states are restored, then reflected for k10. Ground world origins and world root/joint states match the parent first16 or first1 states exactly, so num_envs1 is the same env0 initial sample as num_envs16. Air keeps the configured fixed world anchor and restores matching parent joint states with identity root orientation/zero root velocities. Commands and resets use evaluation seed1 and the exact issue20 ordinals: stand1005, forward0.5=1003. k10 initial root/joint states are reflected with no extra RNG draws. Gait observations use offset10 modulo20; production reward/clock code is unchanged.

Policy counterfactual probe evaluates the same intrinsic actor on exactly transformed observations every step; it is separate from the error between two actual diverging trajectories. Full actor state hashes before/after every job must match.

## First deviation by case

|Case|initial root max|obs t0 max|action t0 max|joint q after1|joint qdot after1|torque after1|first contact post-step|
|---|---:|---:|---:|---:|---:|---:|---|
|air_s1_n1_stand_open|0|8.7422777e-08|0|0.00014817715|0.017061174|0.0012178868|none|
|air_s1_n1_forward_05_open|0|8.7422777e-08|0|0.00020539761|0.017919898|0.001362212|none|
|air_s1_n16_stand_open|0|8.7422777e-08|0|0.00014817715|0.017061651|0.0012179576|none|
|air_s1_n16_forward_05_open|0|8.7422777e-08|0|0.00020539761|0.017920256|0.0013622642|none|
|air_s2_n1_stand_open|0|8.7422777e-08|0|0.00028030248|0.01273632|0.00048233941|none|
|air_s2_n1_forward_05_open|0|8.7422777e-08|0|0.00046405569|0.0070354342|0.00047218055|none|
|air_s2_n16_stand_open|0|8.7422777e-08|0|0.00028030248|0.012736917|0.00048262998|none|
|air_s2_n16_forward_05_open|0|8.7422777e-08|0|0.00046406873|0.0070357323|0.00047270954|none|
|air_s3_n1_stand_open|0|8.7422777e-08|0|8.4400177e-05|0.020730168|0.0018402413|none|
|air_s3_n1_forward_05_open|0|8.7422777e-08|0|0.00010085106|0.019687682|0.0016712379|none|
|air_s3_n16_stand_open|0|8.7422777e-08|0|8.4400177e-05|0.020730525|0.0018403679|none|
|air_s3_n16_forward_05_open|0|8.7422777e-08|0|0.00010085106|0.019687921|0.0016716067|none|
|air_s4_n1_stand_open|0|8.7422777e-08|0|0.00022937264|0.014284611|0.00073904544|none|
|air_s4_n1_forward_05_open|0|8.7422777e-08|0|0.00036764238|0.01007551|0.00055582821|none|
|air_s4_n16_stand_open|0|8.7422777e-08|0|0.00022937637|0.014284849|0.0007391274|none|
|air_s4_n16_forward_05_open|0|8.7422777e-08|0|0.00036764611|0.010076046|0.00055584311|none|
|air_s5_n1_stand_open|0|8.7422777e-08|0|0.00012981892|0.017811418|0.0013414174|none|
|air_s5_n1_forward_05_open|0|8.7422777e-08|0|0.00011849403|0.019003451|0.001548551|none|
|air_s5_n16_stand_open|0|8.7422777e-08|0|0.00012981892|0.017811686|0.0013416074|none|
|air_s5_n16_forward_05_open|0|8.7422777e-08|0|0.00011849403|0.01900357|0.0015488006|none|
|ground_s1_n1_stand_closed|0|8.7422777e-08|7.4505806e-08|0.0009489283|0.04692775|0.011132233|1|
|ground_s1_n1_forward_05_closed|0|8.7422777e-08|5.9604645e-08|0.0013122074|0.20657867|0.050149806|1|
|ground_s1_n1_stand_open|0|8.7422777e-08|0|0.0009489255|0.046927691|0.011132274|1|
|ground_s1_n1_forward_05_open|0|8.7422777e-08|0|0.0013122037|0.20657837|0.050149642|1|
|ground_s1_n16_stand_closed|0|8.7422777e-08|1.3411045e-07|0.0009489283|0.047385991|0.012589242|1|
|ground_s1_n16_forward_05_closed|0|8.7422777e-08|1.1920929e-07|0.0013331547|0.20871586|0.051049992|1|
|ground_s1_n16_stand_open|0|8.7422777e-08|0|0.00094892364|0.047385573|0.012588933|1|
|ground_s1_n16_forward_05_open|0|8.7422777e-08|0|0.0013331585|0.20871574|0.051049985|1|
|ground_s2_n1_stand_closed|0|8.7422777e-08|1.1920929e-07|0.00090988725|0.047491968|0.053593887|1|
|ground_s2_n1_forward_05_closed|0|8.7422777e-08|1.1920929e-07|0.0013100449|0.11372101|0.029228769|1|
|ground_s2_n1_stand_open|0|8.7422777e-08|0|0.00090989284|0.047493517|0.053603299|1|
|ground_s2_n1_forward_05_open|0|8.7422777e-08|0|0.0013100337|0.11372119|0.029228784|1|
|ground_s2_n16_stand_closed|0|8.7422777e-08|1.7881393e-07|0.00092045497|0.049403243|0.055773828|1|
|ground_s2_n16_forward_05_closed|0|8.7422777e-08|1.1920929e-07|0.0013176184|0.11792117|0.029770076|1|
|ground_s2_n16_stand_open|0|8.7422777e-08|0|0.00092052855|0.049403399|0.055796364|1|
|ground_s2_n16_forward_05_open|0|8.7422777e-08|0|0.0013176128|0.11792165|0.029770091|1|
|ground_s3_n1_stand_closed|0|8.7422777e-08|1.1920929e-07|0.00084070954|0.043209955|0.0094773471|1|
|ground_s3_n1_forward_05_closed|0|8.7422777e-08|4.4703484e-08|0.00088943727|0.25175393|0.059643954|1|
|ground_s3_n1_stand_open|0|8.7422777e-08|0|0.00084070954|0.04320991|0.0094774738|1|
|ground_s3_n1_forward_05_open|0|8.7422777e-08|0|0.00088943634|0.2517544|0.059643999|1|
|ground_s3_n16_stand_closed|0|8.7422777e-08|2.3841858e-07|0.00084402226|0.043804571|0.0097883493|1|
|ground_s3_n16_forward_05_closed|0|8.7422777e-08|2.3841858e-07|0.00091210194|0.25516778|0.060204268|1|
|ground_s3_n16_stand_open|0|8.7422777e-08|0|0.00084401574|0.043804467|0.0097886398|1|
|ground_s3_n16_forward_05_open|0|8.7422777e-08|0|0.00091210101|0.25516784|0.060204282|1|
|ground_s4_n1_stand_closed|0|8.7422777e-08|1.1920929e-07|0.00070208311|0.049720779|0.014993645|1|
|ground_s4_n1_forward_05_closed|0|8.7422777e-08|1.1920929e-07|0.00083949603|0.26071656|0.061373249|1|
|ground_s4_n1_stand_open|0|8.7422777e-08|0|0.00070208637|0.049720719|0.014993668|1|
|ground_s4_n1_forward_05_open|0|8.7422777e-08|0|0.00083948486|0.26071674|0.061373249|1|
|ground_s4_n16_stand_closed|0|8.7422777e-08|3.5762787e-07|0.00072910078|0.050304651|0.015798636|1|
|ground_s4_n16_forward_05_closed|0|8.7422777e-08|1.1920929e-07|0.00085722841|0.26277614|0.062627852|1|
|ground_s4_n16_stand_open|0|8.7422777e-08|0|0.00072910124|0.050304905|0.015798606|1|
|ground_s4_n16_forward_05_open|0|8.7422777e-08|0|0.00085722283|0.2627759|0.062627763|1|
|ground_s5_n1_stand_closed|0|8.7422777e-08|1.7881393e-07|0.00099484436|0.042385936|0.013999306|1|
|ground_s5_n1_forward_05_closed|0|8.7422777e-08|5.9604645e-08|0.00120667|0.21962243|0.054065764|1|
|ground_s5_n1_stand_open|0|8.7422777e-08|0|0.00099484436|0.042386144|0.013999343|1|
|ground_s5_n1_forward_05_open|0|8.7422777e-08|0|0.00120667|0.21962225|0.054065652|1|
|ground_s5_n16_stand_closed|0|8.7422777e-08|1.7881393e-07|0.00099484855|0.04616484|0.01399916|1|
|ground_s5_n16_forward_05_closed|0|8.7422777e-08|2.3841858e-07|0.0012327395|0.22286314|0.05524528|1|
|ground_s5_n16_stand_open|0|8.7422777e-08|0|0.0009948439|0.046164811|0.013999138|1|
|ground_s5_n16_forward_05_open|0|8.7422777e-08|0|0.0012327284|0.22286314|0.05524528|1|

[All joint/body/observation component crossings](first_component_crossings.csv), [group first crossings](first_crossings.csv), [all case statistics and per-environment threshold records](pair_analysis.json), and analysis/*_residuals.npz preserve signed errors plus both norms for every group. Closed/open figures show log error growth; exact zeros are displayed at1e-12 only for logarithmic plotting, with unmodified zeros retained in data.

## Frozen interpretation

O has priority: initial observation/action error>1e-5, or air open-loop joint/torque error after the first step>1e-5. N requires the pipeline/air to pass, initial ground error<=1e-5 followed by exponential growth, and small-epsilon flips. H requires air pass, first-contact error>=1e-4, and no epsilon0.2 flips. Other/mixed evidence is reported without forcing an attribution. Growth and flip criteria are frozen in [classification_rules.json](classification_rules.json); thresholds were published before simulation.

### O evidence components

- air_s1_n1_stand_open: joint_pos, step1, Linf 0.000148177147.
- air_s1_n1_stand_open: joint_vel, step1, Linf 0.0170611739.
- air_s1_n1_stand_open: applied_torque, step1, Linf 0.00121788681.
- air_s1_n1_forward_05_open: joint_pos, step1, Linf 0.000205397606.
- air_s1_n1_forward_05_open: joint_vel, step1, Linf 0.017919898.
- air_s1_n1_forward_05_open: applied_torque, step1, Linf 0.001362212.
- air_s1_n16_stand_open: joint_pos, step1, Linf 0.000148177147.
- air_s1_n16_stand_open: joint_vel, step1, Linf 0.0170616508.
- air_s1_n16_stand_open: applied_torque, step1, Linf 0.00121795759.
- air_s1_n16_forward_05_open: joint_pos, step1, Linf 0.000205397606.
- air_s1_n16_forward_05_open: joint_vel, step1, Linf 0.0179202557.
- air_s1_n16_forward_05_open: applied_torque, step1, Linf 0.00136226416.
- air_s2_n1_stand_open: joint_pos, step1, Linf 0.000280302484.
- air_s2_n1_stand_open: joint_vel, step1, Linf 0.0127363205.
- air_s2_n1_stand_open: applied_torque, step1, Linf 0.000482339412.
- air_s2_n1_forward_05_open: joint_pos, step1, Linf 0.000464055687.
- air_s2_n1_forward_05_open: joint_vel, step1, Linf 0.00703543425.
- air_s2_n1_forward_05_open: applied_torque, step1, Linf 0.000472180545.
- air_s2_n16_stand_open: joint_pos, step1, Linf 0.000280302484.
- air_s2_n16_stand_open: joint_vel, step1, Linf 0.0127369165.
- air_s2_n16_stand_open: applied_torque, step1, Linf 0.000482629985.
- air_s2_n16_forward_05_open: joint_pos, step1, Linf 0.000464068726.
- air_s2_n16_forward_05_open: joint_vel, step1, Linf 0.00703573227.
- air_s2_n16_forward_05_open: applied_torque, step1, Linf 0.000472709537.
- air_s3_n1_stand_open: joint_pos, step1, Linf 8.4400177e-05.
- air_s3_n1_stand_open: joint_vel, step1, Linf 0.0207301676.
- air_s3_n1_stand_open: applied_torque, step1, Linf 0.00184024125.
- air_s3_n1_forward_05_open: joint_pos, step1, Linf 0.000100851059.
- air_s3_n1_forward_05_open: joint_vel, step1, Linf 0.0196876824.
- air_s3_n1_forward_05_open: applied_torque, step1, Linf 0.00167123787.
- air_s3_n16_stand_open: joint_pos, step1, Linf 8.4400177e-05.
- air_s3_n16_stand_open: joint_vel, step1, Linf 0.0207305253.
- air_s3_n16_stand_open: applied_torque, step1, Linf 0.00184036791.
- air_s3_n16_forward_05_open: joint_pos, step1, Linf 0.000100851059.
- air_s3_n16_forward_05_open: joint_vel, step1, Linf 0.0196879208.
- air_s3_n16_forward_05_open: applied_torque, step1, Linf 0.00167160667.
- air_s4_n1_stand_open: joint_pos, step1, Linf 0.000229372643.
- air_s4_n1_stand_open: joint_vel, step1, Linf 0.0142846107.
- air_s4_n1_stand_open: applied_torque, step1, Linf 0.000739045441.
- air_s4_n1_forward_05_open: joint_pos, step1, Linf 0.00036764238.
- air_s4_n1_forward_05_open: joint_vel, step1, Linf 0.0100755095.
- air_s4_n1_forward_05_open: applied_torque, step1, Linf 0.000555828214.
- air_s4_n16_stand_open: joint_pos, step1, Linf 0.000229376368.
- air_s4_n16_stand_open: joint_vel, step1, Linf 0.0142848492.
- air_s4_n16_stand_open: applied_torque, step1, Linf 0.000739127398.
- air_s4_n16_forward_05_open: joint_pos, step1, Linf 0.000367646106.
- air_s4_n16_forward_05_open: joint_vel, step1, Linf 0.010076046.
- air_s4_n16_forward_05_open: applied_torque, step1, Linf 0.000555843115.
- air_s5_n1_stand_open: joint_pos, step1, Linf 0.000129818916.
- air_s5_n1_stand_open: joint_vel, step1, Linf 0.0178114176.
- air_s5_n1_stand_open: applied_torque, step1, Linf 0.00134141743.
- air_s5_n1_forward_05_open: joint_pos, step1, Linf 0.000118494034.
- air_s5_n1_forward_05_open: joint_vel, step1, Linf 0.0190034509.
- air_s5_n1_forward_05_open: applied_torque, step1, Linf 0.00154855102.
- air_s5_n16_stand_open: joint_pos, step1, Linf 0.000129818916.
- air_s5_n16_stand_open: joint_vel, step1, Linf 0.0178116858.
- air_s5_n16_stand_open: applied_torque, step1, Linf 0.00134160742.
- air_s5_n16_forward_05_open: joint_pos, step1, Linf 0.000118494034.
- air_s5_n16_forward_05_open: joint_vel, step1, Linf 0.0190035701.
- air_s5_n16_forward_05_open: applied_torque, step1, Linf 0.00154880062.

### All decision checks

```json
{
  "complete": true,
  "O_components": [
    {
      "case": "air_s1_n1_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.0001481771469116211
    },
    {
      "case": "air_s1_n1_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.017061173915863037
    },
    {
      "case": "air_s1_n1_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0012178868055343628
    },
    {
      "case": "air_s1_n1_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.0002053976058959961
    },
    {
      "case": "air_s1_n1_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.01791989803314209
    },
    {
      "case": "air_s1_n1_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0013622120022773743
    },
    {
      "case": "air_s1_n16_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.0001481771469116211
    },
    {
      "case": "air_s1_n16_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.01706165075302124
    },
    {
      "case": "air_s1_n16_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0012179575860500336
    },
    {
      "case": "air_s1_n16_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.0002053976058959961
    },
    {
      "case": "air_s1_n16_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.017920255661010742
    },
    {
      "case": "air_s1_n16_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0013622641563415527
    },
    {
      "case": "air_s2_n1_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.0002803024835884571
    },
    {
      "case": "air_s2_n1_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.012736320495605469
    },
    {
      "case": "air_s2_n1_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.00048233941197395325
    },
    {
      "case": "air_s2_n1_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00046405568718910217
    },
    {
      "case": "air_s2_n1_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.007035434246063232
    },
    {
      "case": "air_s2_n1_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0004721805453300476
    },
    {
      "case": "air_s2_n16_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.0002803024835884571
    },
    {
      "case": "air_s2_n16_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.012736916542053223
    },
    {
      "case": "air_s2_n16_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0004826299846172333
    },
    {
      "case": "air_s2_n16_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.0004640687257051468
    },
    {
      "case": "air_s2_n16_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.007035732269287109
    },
    {
      "case": "air_s2_n16_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0004727095365524292
    },
    {
      "case": "air_s3_n1_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 8.440017700195312e-05
    },
    {
      "case": "air_s3_n1_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.020730167627334595
    },
    {
      "case": "air_s3_n1_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.001840241253376007
    },
    {
      "case": "air_s3_n1_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00010085105895996094
    },
    {
      "case": "air_s3_n1_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.019687682390213013
    },
    {
      "case": "air_s3_n1_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0016712378710508347
    },
    {
      "case": "air_s3_n16_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 8.440017700195312e-05
    },
    {
      "case": "air_s3_n16_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.020730525255203247
    },
    {
      "case": "air_s3_n16_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0018403679132461548
    },
    {
      "case": "air_s3_n16_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00010085105895996094
    },
    {
      "case": "air_s3_n16_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.019687920808792114
    },
    {
      "case": "air_s3_n16_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0016716066747903824
    },
    {
      "case": "air_s4_n1_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00022937264293432236
    },
    {
      "case": "air_s4_n1_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.014284610748291016
    },
    {
      "case": "air_s4_n1_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0007390454411506653
    },
    {
      "case": "air_s4_n1_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.000367642380297184
    },
    {
      "case": "air_s4_n1_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.010075509548187256
    },
    {
      "case": "air_s4_n1_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0005558282136917114
    },
    {
      "case": "air_s4_n16_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00022937636822462082
    },
    {
      "case": "air_s4_n16_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.014284849166870117
    },
    {
      "case": "air_s4_n16_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0007391273975372314
    },
    {
      "case": "air_s4_n16_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00036764610558748245
    },
    {
      "case": "air_s4_n16_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.010076045989990234
    },
    {
      "case": "air_s4_n16_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0005558431148529053
    },
    {
      "case": "air_s5_n1_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00012981891632080078
    },
    {
      "case": "air_s5_n1_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.01781141757965088
    },
    {
      "case": "air_s5_n1_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0013414174318313599
    },
    {
      "case": "air_s5_n1_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00011849403381347656
    },
    {
      "case": "air_s5_n1_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.019003450870513916
    },
    {
      "case": "air_s5_n1_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0015485510230064392
    },
    {
      "case": "air_s5_n16_stand_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00012981891632080078
    },
    {
      "case": "air_s5_n16_stand_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.017811685800552368
    },
    {
      "case": "air_s5_n16_stand_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0013416074216365814
    },
    {
      "case": "air_s5_n16_forward_05_open",
      "quantity": "joint_pos",
      "step": 1,
      "error": 0.00011849403381347656
    },
    {
      "case": "air_s5_n16_forward_05_open",
      "quantity": "joint_vel",
      "step": 1,
      "error": 0.019003570079803467
    },
    {
      "case": "air_s5_n16_forward_05_open",
      "quantity": "applied_torque",
      "step": 1,
      "error": 0.0015488006174564362
    }
  ],
  "air_full_200_step_pass": false,
  "mirrored_input_policy_probe_pass": true,
  "small_epsilon_majority_flip_every_seed": false,
  "epsilon02_no_single_flip_all_seeds": false,
  "ground_first_contact_all_small": false,
  "ground_first_contact_all_large": true,
  "ground_exponential_all_trials": false,
  "counts": {
    "pairs": 60,
    "air_pairs": 20,
    "ground_open_pairs": 20,
    "perturb_cells": 30,
    "first_contact_small": 0,
    "first_contact_large": 170,
    "growth_exponential": 0,
    "growth_trials": 170
  }
}
```

## Attractor perturbation

For the first10 control steps(0.2s), root world-z angular velocity is set to the signed epsilon before simulation. It is a velocity clamp, not repeated additive increments and not an external torque. Five additional independent cold-start epsilon0 controls provide matching unperturbed1024-env eval1 initial-trial slopes, with exact file SHA bindings. These controls have no yaw clamp, are separate from the30 requested signed perturbation cells and are not a parameter scan. Each signed perturbation also starts in a separate fresh process. Reported flips compare per-environment5-20s OLS slope signs. All falls/reset jumps are excluded from full-duration slopes and counted separately; no survivor-only sign count is presented as the full denominator.

|Train seed|epsilon rad/s|positive|negative|zero|censored|flips vs baseline|fraction valid|mean slope deg/s|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|-0.20|0|1024|0|0|0|0|-4.36701|
|1|-0.05|0|1024|0|0|0|0|-4.36189|
|1|-0.01|0|1024|0|0|0|0|-4.35563|
|1|+0.01|0|1024|0|0|0|0|-4.36106|
|1|+0.05|0|1024|0|0|0|0|-4.35929|
|1|+0.20|0|1024|0|0|0|0|-4.35593|
|2|-0.20|1020|4|0|0|8|0.0078125|0.782553|
|2|-0.05|1022|2|0|0|6|0.00585938|0.779639|
|2|-0.01|1022|2|0|0|6|0.00585938|0.783034|
|2|+0.01|1024|0|0|0|4|0.00390625|0.778938|
|2|+0.05|1023|1|0|0|5|0.00488281|0.761628|
|2|+0.20|1023|1|0|0|5|0.00488281|0.772634|
|3|-0.20|1024|0|0|0|0|0|3.17358|
|3|-0.05|1024|0|0|0|0|0|3.21545|
|3|-0.01|1024|0|0|0|0|0|3.1764|
|3|+0.01|1024|0|0|0|0|0|3.19214|
|3|+0.05|1024|0|0|0|0|0|3.18635|
|3|+0.20|1024|0|0|0|0|0|3.15153|
|4|-0.20|0|1024|0|0|0|0|-1.20874|
|4|-0.05|0|1024|0|0|0|0|-1.20025|
|4|-0.01|0|1024|0|0|0|0|-1.20414|
|4|+0.01|0|1024|0|0|0|0|-1.20126|
|4|+0.05|0|1024|0|0|0|0|-1.21493|
|4|+0.20|0|1024|0|0|0|0|-1.21089|
|5|-0.20|1024|0|0|0|0|0|7.86146|
|5|-0.05|1024|0|0|0|0|0|7.8776|
|5|-0.01|1024|0|0|0|0|0|7.87644|
|5|+0.01|1024|0|0|0|0|0|7.88134|
|5|+0.05|1024|0|0|0|0|0|7.84587|
|5|+0.20|1024|0|0|0|0|0|7.87109|

Full per-environment slopes and exact baseline bindings: [perturbation_analysis.json](perturbation_analysis.json). A finite set of yaw perturbations cannot prove there is mathematically only one attractor; an absence of observed flips is reported within this tested perturbation range.

## Figures and robot videos

- [closed_stand_n16](figures/closed_stand_n16.png)
- [closed_forward_05_n16](figures/closed_forward_05_n16.png)
- [closed_stand_n1](figures/closed_stand_n1.png)
- [closed_forward_05_n1](figures/closed_forward_05_n1.png)
- [open_stand_n16](figures/open_stand_n16.png)
- [open_forward_05_n16](figures/open_forward_05_n16.png)
- [open_stand_n1](figures/open_stand_n1.png)
- [open_forward_05_n1](figures/open_forward_05_n1.png)
- [actuator_aux_stand](figures/actuator_aux_stand.png)
- [actuator_aux_forward_05](figures/actuator_aux_forward_05.png)
- [perturbation_slopes](figures/perturbation_slopes.png)
- [stand k0 front](videos/video_stand_k0_front.mp4)
- [stand k0 side](videos/video_stand_k0_side.mp4)
- [stand k10 front](videos/video_stand_k10_front.mp4)
- [stand k10 side](videos/video_stand_k10_side.mp4)
- [forward_05 k0 front](videos/video_forward_05_k0_front.mp4)
- [forward_05 k0 side](videos/video_forward_05_k0_side.mp4)
- [forward_05 k10 front](videos/video_forward_05_k10_front.mp4)
- [forward_05 k10 side](videos/video_forward_05_k10_side.mp4)

Robot representatives use fixed training seed1/evaluation seed1/env0 from16-environment closed-loop runs for both commands and phases. They are not maximum-trial selections. Exact saved poses are replayed; display velocities are zero, with no dynamics integration. Front and side follow translation and heading. Colors are loaded from a separately saved precolored layer of the exact strict physical USD, never bound after initialization. [Joint colors](joint_colors.json). HumanML3D NPY: N/A, these are stage1 velocity commands, not retargeted motions. Automated encoding/color/frame checks and actual visual review are separate receipts.

## Provenance and limitations

Issue20 static mass/COM/inertia/collision and runtime PhysX checks are existing evidence, not proof of dynamic solver equivariance. Diagnostics isolate timing and components but do not establish hardware causality. Fixed-root/no-contact differs deliberately from floating-base contact mechanics. All required seed and environment-count results remain individually visible.

The first five completed instrumentation-v1 jobs and the interrupted sixth are retained separately after discovering the implicit PD torque estimate semantics. The prescribed queue is rerun with measured PhysX projected active joint force and separate PD estimate/explicit input/target records; thresholds, policies and physical parameters are unchanged. Sequential-process v2 data is also preserved separately after correcting possible prior-trajectory solver-cache contamination; final cold-start sampling uses exact parent initial states and independent processes per command/phase/condition, plus five zero-perturbation controls. See cold_start_sampling_correction.json and torque_instrumentation_correction.json for the separate computation ledger. The initial startup dict-container correction occurred before any simulation step or completed trial; logs and a correction ledger are retained. No failed computation is silently merged into scientific denominators. Historical issue17/19 failures and issue20 required stop remain preserved. SuperMimic stays stopped. Work remains on the remote Linux server, with no mounted-file copy to the local computer. Project records use Notion; no Linear synchronization.

## Concrete O-trigger components and comparison limits

All component/env exceedances at the exact O threshold1e-5 are retained in [O_trigger_components.json](O_trigger_components.json). Maximal triggering component per case/quantity:

|Case|Quantity|Step|Environment|Component|Signed residual|
|---|---|---:|---:|---|---:|
|air_s1_n1_stand_open|joint_pos|1|0|left_shoulder_roll_joint|-0.000148177147|
|air_s1_n1_stand_open|joint_vel|1|0|neck_linkage_roll|-0.0170611739|
|air_s1_n1_stand_open|applied_torque|1|0|waist_roll_joint|-0.00121788681|
|air_s1_n1_forward_05_open|joint_pos|1|0|left_shoulder_roll_joint|-0.000205397606|
|air_s1_n1_forward_05_open|joint_vel|1|0|neck_linkage_roll|-0.017919898|
|air_s1_n1_forward_05_open|applied_torque|1|0|waist_roll_joint|-0.001362212|
|air_s1_n16_stand_open|joint_pos|1|0|left_shoulder_roll_joint|-0.000148177147|
|air_s1_n16_stand_open|joint_vel|1|15|neck_linkage_roll|-0.0170616508|
|air_s1_n16_stand_open|applied_torque|1|13|waist_roll_joint|-0.00121795759|
|air_s1_n16_forward_05_open|joint_pos|1|0|left_shoulder_roll_joint|-0.000205397606|
|air_s1_n16_forward_05_open|joint_vel|1|1|neck_linkage_roll|-0.0179202557|
|air_s1_n16_forward_05_open|applied_torque|1|15|waist_roll_joint|-0.00136226416|
|air_s2_n1_stand_open|joint_pos|1|0|neck_linkage_roll|-0.000280302484|
|air_s2_n1_stand_open|joint_vel|1|0|neck_linkage_roll|-0.0127363205|
|air_s2_n1_stand_open|applied_torque|1|0|waist_roll_joint|-0.000482339412|
|air_s2_n1_forward_05_open|joint_pos|1|0|neck_linkage_roll|-0.000464055687|
|air_s2_n1_forward_05_open|joint_vel|1|0|neck_linkage_roll|-0.00703543425|
|air_s2_n1_forward_05_open|applied_torque|1|0|waist_roll_joint|+0.000472180545|
|air_s2_n16_stand_open|joint_pos|1|3|neck_linkage_roll|-0.000280302484|
|air_s2_n16_stand_open|joint_vel|1|7|neck_linkage_roll|-0.0127369165|
|air_s2_n16_stand_open|applied_torque|1|0|waist_roll_joint|-0.000482629985|
|air_s2_n16_forward_05_open|joint_pos|1|4|neck_linkage_roll|-0.000464068726|
|air_s2_n16_forward_05_open|joint_vel|1|4|neck_linkage_roll|-0.00703573227|
|air_s2_n16_forward_05_open|applied_torque|1|15|waist_roll_joint|+0.000472709537|
|air_s3_n1_stand_open|joint_pos|1|0|left_shoulder_roll_joint|-8.4400177e-05|
|air_s3_n1_stand_open|joint_vel|1|0|neck_linkage_roll|-0.0207301676|
|air_s3_n1_stand_open|applied_torque|1|0|waist_roll_joint|-0.00184024125|
|air_s3_n1_forward_05_open|joint_pos|1|0|left_shoulder_roll_joint|-0.000100851059|
|air_s3_n1_forward_05_open|joint_vel|1|0|neck_linkage_roll|-0.0196876824|
|air_s3_n1_forward_05_open|applied_torque|1|0|waist_roll_joint|-0.00167123787|
|air_s3_n16_stand_open|joint_pos|1|0|left_shoulder_roll_joint|-8.4400177e-05|
|air_s3_n16_stand_open|joint_vel|1|8|neck_linkage_roll|-0.0207305253|
|air_s3_n16_stand_open|applied_torque|1|14|waist_roll_joint|-0.00184036791|
|air_s3_n16_forward_05_open|joint_pos|1|1|left_shoulder_roll_joint|-0.000100851059|
|air_s3_n16_forward_05_open|joint_vel|1|13|neck_linkage_roll|-0.0196879208|
|air_s3_n16_forward_05_open|applied_torque|1|11|waist_roll_joint|-0.00167160667|
|air_s4_n1_stand_open|joint_pos|1|0|neck_linkage_roll|-0.000229372643|
|air_s4_n1_stand_open|joint_vel|1|0|neck_linkage_roll|-0.0142846107|
|air_s4_n1_stand_open|applied_torque|1|0|waist_roll_joint|-0.000739045441|
|air_s4_n1_forward_05_open|joint_pos|1|0|neck_linkage_roll|-0.00036764238|
|air_s4_n1_forward_05_open|joint_vel|1|0|neck_linkage_roll|-0.0100755095|
|air_s4_n1_forward_05_open|applied_torque|1|0|right_shoulder_roll_joint|-0.000555828214|
|air_s4_n16_stand_open|joint_pos|1|1|neck_linkage_roll|-0.000229376368|
|air_s4_n16_stand_open|joint_vel|1|7|neck_linkage_roll|-0.0142848492|
|air_s4_n16_stand_open|applied_torque|1|2|waist_roll_joint|-0.000739127398|
|air_s4_n16_forward_05_open|joint_pos|1|6|neck_linkage_roll|-0.000367646106|
|air_s4_n16_forward_05_open|joint_vel|1|3|neck_linkage_roll|-0.010076046|
|air_s4_n16_forward_05_open|applied_torque|1|2|right_shoulder_roll_joint|-0.000555843115|
|air_s5_n1_stand_open|joint_pos|1|0|left_shoulder_roll_joint|-0.000129818916|
|air_s5_n1_stand_open|joint_vel|1|0|neck_linkage_roll|-0.0178114176|
|air_s5_n1_stand_open|applied_torque|1|0|waist_roll_joint|-0.00134141743|
|air_s5_n1_forward_05_open|joint_pos|1|0|left_shoulder_roll_joint|-0.000118494034|
|air_s5_n1_forward_05_open|joint_vel|1|0|neck_linkage_roll|-0.0190034509|
|air_s5_n1_forward_05_open|applied_torque|1|0|waist_roll_joint|-0.00154855102|
|air_s5_n16_stand_open|joint_pos|1|0|left_shoulder_roll_joint|-0.000129818916|
|air_s5_n16_stand_open|joint_vel|1|13|neck_linkage_roll|-0.0178116858|
|air_s5_n16_stand_open|applied_torque|1|6|waist_roll_joint|-0.00134160742|
|air_s5_n16_forward_05_open|joint_pos|1|1|left_shoulder_roll_joint|-0.000118494034|
|air_s5_n16_forward_05_open|joint_vel|1|9|neck_linkage_roll|-0.0190035701|
|air_s5_n16_forward_05_open|applied_torque|1|12|waist_roll_joint|-0.00154880062|

Each command/phase/condition starts in a separate fresh process. Both environment counts restore the same parent issue20 initial sample; num_envs1 is the exact env0 initial state of the16-env experiment (ground world origins also match). Air uses the same matched joint state and its configured fixed anchor. Thus no preceding phase or command can supply trajectory-specific solver/contact history. Cold application initialization and unavoidable floating-point solver behavior remain part of the tested simulator.

Pure implicit-drive torque is not exposed separately by the installed Isaac Lab actuator API. PhysX projected active joint force, approximate PD estimate and explicit actuation input are separately retained and labeled; no estimate is presented as a direct motor measurement. Classification uses the measured active joint component together with joint state errors under the frozen thresholds.

Ground-open fall flags are retained while automatic reset requests are suppressed: all340 final ground-open trajectories continue physically for200 control steps from one initial state, even after a fall condition. Original termination tests, physical parameters and saved actions are unchanged. Independent episode-counter and source-hash checks are recorded in independent_open_continuity_audit.json. The earlier reset-contaminated40 replays are separately retained; see ground_open_lifecycle_correction.json.
Instrumentation-v1 and startup/queue orchestration failures are retained with distinct ledgers. Only the25 final logical groups contribute to required scientific group counts; five unperturbed controls are separately reported. The active final cold plan contains155 processes; total executed cold processes195 includes40 rejected reset-contaminated ground-open replays, all separately accounted. Runtime hashes and original thresholds stay fixed; analysis/presentation freeze is documented separately.

## Stage localization and perturbation interpretation

The complete five-seed evidence localizes the earliest observed deviation as follows:

- Initial closed-loop observation maximum component residual: 8.74227766e-08; action residual: 3.57627869e-07. Exact-transformed-input actor probe maximum: 0.
- Air/no-contact first-step residual ranges across all paired conditions: {"joint_pos": [8.440017700195312e-05, 0.0004640687257051468], "joint_vel": [0.007035434246063232, 0.020730525255203247], "applied_torque": [0.0004721805453300476, 0.0018403679132461548], "joint_target": [0.0, 0.0]}.
- Air maximum contact force: 0N; mirrored position-target error over all200 steps: 0rad.

These observations meet the frozen O rule. When the mirrored position targets are identical and contact forces are zero, the observed joint-state/active-force mismatch arises in the no-contact plant or its mirror coordinate contract, rather than requiring contact or a policy feedback difference. This does not isolate a defective source line or prove a particular PhysX, inertia or joint-frame cause. Static parameter symmetry alone cannot certify full nonlinear kinematic/drive/solver equivariance. The exact offending joint names, signs, environments and steps are reported above.

Perturbation comparison uses five separate cold unperturbed controls. Per training seed:

- Seed1 baseline +0/-1024, censored0: eps-0.20: +0/-1024/zero0/censored0, flips0 of1024 compared; eps-0.05: +0/-1024/zero0/censored0, flips0 of1024 compared; eps-0.01: +0/-1024/zero0/censored0, flips0 of1024 compared; eps+0.01: +0/-1024/zero0/censored0, flips0 of1024 compared; eps+0.05: +0/-1024/zero0/censored0, flips0 of1024 compared; eps+0.20: +0/-1024/zero0/censored0, flips0 of1024 compared.
- Seed2 baseline +1020/-4, censored0: eps-0.20: +1020/-4/zero0/censored0, flips8 of1024 compared; eps-0.05: +1022/-2/zero0/censored0, flips6 of1024 compared; eps-0.01: +1022/-2/zero0/censored0, flips6 of1024 compared; eps+0.01: +1024/-0/zero0/censored0, flips4 of1024 compared; eps+0.05: +1023/-1/zero0/censored0, flips5 of1024 compared; eps+0.20: +1023/-1/zero0/censored0, flips5 of1024 compared.
- Seed3 baseline +1024/-0, censored0: eps-0.20: +1024/-0/zero0/censored0, flips0 of1024 compared; eps-0.05: +1024/-0/zero0/censored0, flips0 of1024 compared; eps-0.01: +1024/-0/zero0/censored0, flips0 of1024 compared; eps+0.01: +1024/-0/zero0/censored0, flips0 of1024 compared; eps+0.05: +1024/-0/zero0/censored0, flips0 of1024 compared; eps+0.20: +1024/-0/zero0/censored0, flips0 of1024 compared.
- Seed4 baseline +0/-1024, censored0: eps-0.20: +0/-1024/zero0/censored0, flips0 of1024 compared; eps-0.05: +0/-1024/zero0/censored0, flips0 of1024 compared; eps-0.01: +0/-1024/zero0/censored0, flips0 of1024 compared; eps+0.01: +0/-1024/zero0/censored0, flips0 of1024 compared; eps+0.05: +0/-1024/zero0/censored0, flips0 of1024 compared; eps+0.20: +0/-1024/zero0/censored0, flips0 of1024 compared.
- Seed5 baseline +1024/-0, censored0: eps-0.20: +1024/-0/zero0/censored0, flips0 of1024 compared; eps-0.05: +1024/-0/zero0/censored0, flips0 of1024 compared; eps-0.01: +1024/-0/zero0/censored0, flips0 of1024 compared; eps+0.01: +1024/-0/zero0/censored0, flips0 of1024 compared; eps+0.05: +1024/-0/zero0/censored0, flips0 of1024 compared; eps+0.20: +1024/-0/zero0/censored0, flips0 of1024 compared.

These are empirical sign changes within the prescribed perturbation range, not a proof of the global number of attractors. O takes precedence over N/H under the frozen rules; the perturbation data does not override the detected no-contact failure. No corrective training or default change is implemented. The single next-step recommendation remains the one stated at the beginning of this report.
