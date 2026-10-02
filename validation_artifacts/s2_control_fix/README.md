# S2 control fix final candidates

Selected candidate: **acc_train**. Both trials warm-start S2 smooth, 600 additional PPO iterations, seed42, 1024 environments. Joint target velocity is capped at 50% actuator limits, with unchanged PD gains and no target low-pass. acc_train uses acceleration penalty -2e-6; rate_train uses -1e-7.

All 7 acc_train clips completed without abnormal termination in deterministic seed42 evaluation. rate_train had two end-effector termination events on weight-shift and is not selected. Multi-seed/noise robustness is not yet accepted.

|Walking physical 2-8 Hz RMS|Before|After|Reduction|
|---|---:|---:|---:|
|All joints pooled, degrees|1.944|1.417|27.1%|
|right_hip_roll_joint|4.332|2.009|53.6%|
|left_hip_pitch_joint|4.126|2.174|47.3%|
|left_hip_roll_joint|3.973|2.118|46.7%|
|gearbox_roll|3.458|1.385|60.0%|
|left_knee_pitch_joint|2.851|1.946|31.7%|
|chest_pitch|2.680|1.337|50.1%|

Walking body-position tracking error (unweighted mean over 4 clip metrics): reference_smooth=0.01521 m, acc_train=0.01954 m.

Limitations: oscillation is reduced, not eliminated. Some ankle/yaw joints worsen; see all-joint plot and JSON. Stand RMS increases from .084 to .246 degrees, weight-shift .263 to .468, wave .288 to .276. Complete motion-end alone does not establish faithful imitation or motor feasibility.

Camera follows low-pass horizontal translation only (tau=.5s), with fixed height and direction. All continuous segments have zero camera vertical range. Camera traces are included; reset cuts can occur. Videos: H264, 1280x720, 50fps, 999 frames (~20s). Physical jitter above was computed from telemetry, independently of rendering.

Keep each final model_11597.pt alongside its params/control_runtime.json; playback restores the target-rate behavior. External deployment must implement the same rate limiter. Intermediate checkpoints are intentionally excluded.

Plots exclude resets and bandpass edge samples; actual angles sampled at200Hz, targets at50Hz. Band 2-8Hz captures the observed ~4.3Hz oscillation and is not a universal perceptual threshold.
