# TW-64: S1 final validation

Training completed on 2026-10-02 around 17:01 Asia/Shanghai at iteration 9999
(10000 PPO updates). Final checkpoint:
`logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d/2026-10-02_14-01-51_tw59_s1_v2/model_9999.pt`.
Checkpoint: 6861173 bytes; all 68 checkpoint tensors finite, load/inference passed.

The first per-clip replay hit the adaptive sampling cap at the feasibility
boundary. Evaluation now uses the existing fixed-motion override, frame zero
resets and no adaptive probability cap. Training settings and weights are unchanged.
Replay writes termination counts and mean motion errors via --evaluation_output.
Rendering can be retried without training using scripts/practice9/render_tw59_s1.sh.
80 existing Python tests passed; headless real replay succeeded on all four clips.

| Motion | Replay steps | Video seconds | Mean body position error (cm) | Abnormal terminations |
| --- | ---: | ---: | ---: | ---: |
| 000801 | 1000 | 19.98 | 1.685 | 0 |
| 006680 | 1250 | 24.98 | 1.321 | 0 |
| 010407 | 1000 | 19.98 | 1.776 | 0 |
| 010782 | 1000 | 19.98 | 1.519 | 0 |

Each clip reaches its normal motion_end at least once; no anchor position,
anchor orientation or end-effector position termination occurred in these runs.
Videos are H.264, 1280x720, 50fps, fully decoded without error. Original videos,
zoomed copies, per-clip metric JSON and validation_summary.json stay under
`/root/gpufree-data/datasets/practice9/tw59_s1/` on the remote server.
Contact-sheet inspection shows upright walking/posture tracking.

This is one replay per training clip, not a held-out or repeated-seed evaluation.
Playback retained configured environment randomization and did not fix a seed.
Low body-position error does not establish exact joint/velocity tracking, sim-to-real
transfer, or broad robustness. The data remains the four retargeted HumanML3D clips
(3746 frames at 50Hz), not the full dataset.
