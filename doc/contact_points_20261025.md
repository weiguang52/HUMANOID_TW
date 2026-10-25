# Contact-point diagnostic: TW-RL / Notion004

2026-10-25 12:20 Asia/Shanghai. User authorized continued optimization until acceptance.
Goal: distinguish upper-motion foot lifting/rolling/sliding before changing a reward.
Frozen original27995 and baseline28494, synthetic static2/upper3, seed42,20s;
no disturbance or training. Colored head-locked asset; side and translation-following
front views. State: /root/gpufree-data/datasets/practice9/contact_points_20261025.

Opt-in P9_CONTACT_POINT_TELEMETRY adds a separate sensor for each foot, filtered
against the verified /World/ground/terrain/GroundPlane/CollisionPlane prim.
A wildcard initially returned zero points despite a valid filter count; smoke_v4
with exact collision path returned up to4points/foot and120/120finite samples.
Production diagnostics select only points with positive normal force >1e-6N,
record counts, total normal load, mean point and RMS worldXY rigid-point velocity.
Velocity uses v_COM + omega cross(point-COM); stationary ground so this is relative
tangential speed. Existing body_lin_vel_w is COM velocity (earlier reports called
it foot-link speed); neither COM speed nor link-origin speed equals contact slip.
Physics contacts and post-step state are discrete samples, not continuous ground truth.

Three checks passed: translation, rest and pure rolling cancellation. Syntax passed.
Initial short smoke was rejected by the strict20s perturbation protocol; that flag
was removed since this diagnostic has no disturbance. All failed outputs retained.
The frozen side jobs started before helper extraction; front jobs use the equivalent
rigid_point_velocity helper. No physical parameters changed in that refactor.
Before inference: check point coverage, force agreement, buffer capacity, and trajectory
agreement with previous zero-disturbance runs. No training or promotion until evidence.

Acceptance for subsequent optimization: static drift below109.15mm and matched
baseline, upper20s completion without drift regression, standing6/6, walking>=23/24,
turning3/3; preserve per-group results and verify videos. Synthetic diagnostics have
no original HumanML3D source clip; do not relabel historical source clips as their reference.
