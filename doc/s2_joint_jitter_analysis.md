# Walking joint-command oscillation analysis

Data: four walking clips (000801,006680,010407,010782), S2 baseline and smooth,
final model_10998.pt, seed42, observation corruption disabled. Commands are
physical target joint angles, not normalized neural-network outputs. Plots show
all 30 joints, reference/command/actual angles, and two-second detailed traces.

Reset steps and adjacent steps are removed. A fourth-order zero-phase 2-8Hz
bandpass or >8Hz highpass is applied independently to continuous valid segments;
0.5s is trimmed at each filter edge. Commands/reference are sampled at50Hz and
actual angles at200Hz. RMS is pooled over the four clips. These frequency bands
are diagnostic choices, not definitions of all unwanted motion.

The dominant visible oscillation was missed by the earlier >8Hz-only summary.
In clip000801, all five leading joints show a dominant 2-8Hz-band peak near
4.30Hz (Welch resolution 0.195Hz). A shared closed-loop oscillation is consistent
with the data; it is not proof of a specific PD/latency mechanism. Observation
noise is disabled, and the reference has almost no power in this band.

|Joint|Smooth command RMS deg|Actual RMS deg|Reference RMS deg|Baseline command RMS deg|
|---|---:|---:|---:|---:|
|right_hip_roll_joint|7.49|4.33|0.021|12.36|
|gearbox_roll|7.16|3.46|0.010|16.40|
|left_hip_pitch_joint|6.89|4.13|0.052|14.22|
|left_hip_roll_joint|6.83|3.97|0.023|12.02|
|left_knee_pitch_joint|4.95|2.85|0.114|9.51|
|chest_pitch|4.69|2.68|0.007|13.57|

These are 2-8Hz RMS amplitudes, not peak-to-peak ranges. Smooth helps substantially
but leaves multi-joint hip/torso oscillations; there is no single isolated bad
joint. In the zoomed right-hip-roll trace, command oscillations still reach
roughly +/-10 degrees about the slower trend. Also note reference mismatch:
left knee reference flexion reaches about -70 degrees in clip000801, whereas
actual flexion is much smaller. Completing motion_end is not faithful imitation.

Next controlled checks: target the hip/torso groups; vary PD gains or introduce
a modest command-filter/rate constraint one factor at a time, within training
and evaluation consistently. Measure 2-8Hz absolute RMS, jerk/torque, tracking
and completion, not just normalized >8Hz power. Do not blindly increase all
joint smoothness penalties or apply a strong deployment-only filter.

Side videos: both variants replay clip000801 from a following camera at
(0,-1.2,0.15), looking at (0,0,-0.03) relative to the root. 1280x720 H264,
50fps,999frames (~20s). Preview inspected for correct side view/full body.
The camera is the only requested display change; physical replay stays enabled.

All plots, side videos, final checkpoints, configs and SHA256 manifest are in
validation_artifacts/s2_joint_analysis. Raw telemetry remains on the server.
