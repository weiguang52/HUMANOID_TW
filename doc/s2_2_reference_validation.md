# TW-67: simple references and dynamics audit

Mass is 0.840743 kg, confirmed by the user. All 33 inertial links passed positive
mass, positive-definite inertia and principal-moment triangle checks in the
baseline audit. Joint-coordinate conversion/native mapping: 12 tests pass.
Keep tw44_table_v2 and both knee/shoulder pitch hard limits at +/-90 degrees.
Feet use the URDF collision boxes (about 49 x 22.85 x 20.94 mm). Base is free,
self-collision is disabled. PD uses implicit actuators; contact is physical.

Current candidate groups (effort Nm, velocity rad/s, stiffness, damping):
legs (40,3,20,1), feet (15,3,10,.5), waist (20,2,10,.5), arms (8,2,5,.25),
neck (3,2,3,.15). These are bootstrap simulation parameters, not selected motors.
The 20 walking runs' measured per-joint torque/speed envelope is recorded in
s2_motor_load_baseline.json. For example left hip pitch RMS estimate is 3.43 Nm,
p99 7.21 Nm, peak 10.81 Nm. Current shaking inflates demand. Implicit actuator
estimates and transient speed-limit overshoot are not motor specification proof.
Reduce shaking before motor sizing, then model torque-speed curves, gearbox
losses, thermal duty, rotor inertia and delays. No mass/limit changes made here.

The initial 210 nonwalking candidates remain rejected. The selected wave's
only >5% near-limit occupancy was neck roll (6.7%). Preserve its contiguous
interior frames [135:809], 674 frames at 50 Hz, rather than modifying its angles.
This is an explicitly cropped wave, not a successful replay of the full source.

Three new references pass the unchanged position/velocity/time-scaling gates
and Isaac FK readback/velocity checks:
- s2_stand: 501 frames; explicit procedural stance, arms initialized from the
  source wave's final frame, neutral head, leg IK on the training URDF.
- s2_weight_shift: 501 frames; 8 mm lateral pelvis motion with 5 s period and
  planted-foot position/orientation constraints.
- s2_wave_000113: 674-frame unmodified joint-trajectory crop described above.

Independent Pinocchio FK/support checks: stand COM projection margin 16.87 mm;
weight shift minimum 15.98 mm and maximum foot-center coordinate range .888 mm.
These are geometric feasibility checks, not closed-loop stability guarantees.
FK conversion grounds the references using actual foot collision corners.
Source hashes, derivations, quality checks and ground corrections are in
s2_simple_reference_audit.json. Files stay in datasets/practice9/s2_simple_v1.
Native-only pipeline admission remains the default; the S2 trial launcher
explicitly admits the validated mixed curriculum backend and retains all
coordinate, order, FPS, quality, file-existence and FK gates.

S1 fixed-seed physical replays each ran 1000 steps without observation noise:
stand anchor/endpoint flags 11/7; shift 7/8; wave 1/14; all motion_end=0.
Flags can overlap. S1 does not perform these actions reliably; retain these as
failure baselines. TW-67 delivers usable references, not a trained S2 policy.

Next (TW-68): two independent GPU trials, each starting model_9999.pt, seed42,
1024 environments, 1000 additional PPO iterations. Same 7-reference curriculum;
only action-rate reward differs: -.05 versus -.2. Walking weights are 1 each,
stand 1, shift/wave .5 each. Do not infer 2x single-policy speedup from two jobs.
Only accept improvements after physical replay and no walking regression.
