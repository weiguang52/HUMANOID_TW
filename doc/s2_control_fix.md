# S2 control oscillation and camera correction

The 4.3 Hz oscillation is a closed-loop problem, so adding a low-pass filter
without physical replay is not sufficient. Keep the unmodified S2 smooth model
as the reference. All initial probes use clip000801/seed42/no observation noise.
Actual-joint 2-8Hz RMS (averaged over joints/time) was 1.816 degrees originally.

|Change|Actual RMS deg|Complete without abnormal termination|
|---|---:|---|
|Target low-pass tau=.04s|2.267|yes|
|Target rate capped at actuator velocity limit|1.571|yes|
|Leg/waist damping x.5|2.343|yes|
|Leg/waist stiffness x.5|1.405|no|
|Leg/waist damping x2|1.349|no|
|Leg/waist stiffness x.75|1.429|yes|
|Target rate capped at 50% actuator velocity|1.443|yes|
|Stiffness x.75 plus 50% target rate|1.366|no|

Lower RMS alone is not acceptance: falling/short episodes can distort statistics.
Rejected probes are retained only as diagnostics. Existing global gains stay
unchanged. The selected training constraint is a per-joint target rate limit,
50% of configured actuator velocity; no additional low-pass latency is adopted.
Two 600-iteration warm-start trials compare original versus 20x joint-acceleration
penalty (-1e-7 versus -2e-6); action-rate penalty stays -.2. Final rollout
comparison is required before selecting a result.

Implementation: SmoothJointPositionAction shares processing across train/play.
The state resets only for the affected environments and is initialized from
actual joint positions after reset events. Disabled shaping preserves original
behavior. Raw policy actions remain unchanged for observations and penalties.
Control units are radians and radians/second. Optional PD multipliers and tau
remain explicit experiment knobs, not silent changes to old checkpoints.

Training writes params/control_runtime.json with target and PD settings.
Playback automatically restores it, unless an explicit experiment environment
variable overrides a field. Keep this file with exported weights. A hardware
or external simulator controller must implement the same target-rate operation;
loading the neural-network weights alone does not reproduce this controller.

Camera: follow only filtered horizontal translation (tau=.5s), keep the initial
height, and use a constant viewing direction. Recenter on an episode reset.
Camera motion is separate from physics and is saved as camera/robot origins in
a .camera.npz file alongside evaluation JSON. This avoids translating the
camera with each root-height oscillation. Reset cuts are still expected.

Validation: 87 practice9 tests and 8 subtests pass; new tests cover target-speed
bounds/joint ordering/partial resets, unchanged disabled behavior, saved runtime
restoration, and camera suppression of vertical/fast horizontal motion.

Final validation selects acc_train provisionally: all seven seed42 clips complete
without abnormal termination. rate_train fails the weight-shift gate (two
end-effector terminations). Four-clip pooled physical 2-8Hz RMS falls from 1.944
to 1.417 degrees (27.1%). Right hip roll improves 53.6%, left hip pitch47.3%,
left hip roll46.7%, gearbox roll60.0%, left knee31.7%, chest50.1%.
Mean walking body-position error increases from .01521m to .01954m. Some ankle
and yaw joints worsen; stand/weight-shift RMS also increases. This is a walking
oscillation improvement, not global acceptance or complete elimination.

Both final checkpoints and stable-camera walking videos are in
validation_artifacts/s2_control_fix, with runtime configs, evaluations, plots,
and SHA256 hashes. Camera telemetry confirms zero within-segment height change;
selected camera XY band RMS is .225mm versus robot XY2.082mm (89.2% suppression).
Both videos verify H264/720p/50fps/999frames. Original weights remain available.
TW69 multi-seed/noise robustness remains outstanding before promotion.
