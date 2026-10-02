# S2: 4000-iteration controlled continuation

Both trials start from acc_train/model_11597.pt, use seed42 and1024env. long_control preserves the objective; long_velocity adds mean squared reference-joint velocity error with weight -0.2. PD, target rate (.5 actuator speed), acceleration and action-rate penalties remain unchanged. Each final checkpoint is model_15596.pt.

|Candidate|Walking physical 2-8Hz RMS (deg)|Mean body-position error (m)|
|---|---:|---:|
|smooth|1.944|0.01521|
|acc_train|1.417|0.01954|
|long_control|1.047|0.02172|
|long_velocity|0.866|0.02098|

All 30 joint comparisons are in joint_comparison.png. Lower average oscillation does not prove all joints improved or faithful imitation.

## Robustness failures

Each candidate has 28 replays: seven clips times seed42/123/2026 without observation noise, plus seed123 with noise. These are replay seeds, not independent training seeds. Each replay lasts at least1000steps or one motion length; reset segments may follow failures. Completion after reset does not erase a failure.

long_control: 3/28 replays contain abnormal termination.
- seed42/010782: {'ee_body_pos': 1}
- seed123/010782: {'ee_body_pos': 1}
- seed2026/010782: {'ee_body_pos': 2}
long_velocity: 3/28 replays contain abnormal termination.
- seed123/006680: {'ee_body_pos': 1}
- seed2026/010782: {'anchor_pos': 1}
- noise123/006680: {'ee_body_pos': 1}

Neither candidate is promoted as fully accepted. long_velocity is the smoother candidate; full stability remains unresolved. Previous weights are preserved. No motor specification is inferred from these results.

## Camera and artifacts

Camera height/direction fixed, horizontal tracking tau=.5s. Within-segment height range is zero (see summary.json); episode resets can cut. Side videos are H2641280x720/50fps999frames. Joint oscillation is measured from physical telemetry independently of camera motion.

Keep final weights beside params/control_runtime.json. Replay restores target shaping; deployment must implement the same rate limiter. No intermediate checkpoints included. Server shutdown interrupted only robustness evaluation; resume_s2_long_validation.sh verifies existing JSON/telemetry and completes missing outputs.
