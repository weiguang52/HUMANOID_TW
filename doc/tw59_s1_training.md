# TW-59 S1 training (2026-10-02)

Remote tmux session: `tw59_s1`.
Launcher: `bash scripts/practice9/run_tw59_s1.sh`.
State and logs: `/root/gpufree-data/datasets/practice9/tw59_s1`.
GPU 1, 1024 environments, seed 42, 10000 PPO iterations, run `tw59_s1_v2`.
Starting from scratch because pre-correction policies use different action
coordinates/limits; the current-coordinate smoke checkpoint has only one iteration.

Data: `/root/gpufree-data/datasets/practice9/tw56_native_v1/manifest.json`,
contract `tw44_table_v2`. Four validated HumanML3D motions: 000801, 006680,
010407, 010782, totaling 3746 frames at 50 Hz (74.92 seconds after retiming).
These were selected from 165 previous locomotion candidates using the native
C++ retargeter, joint-limit/dynamic gates and simulator FK validation.
The training set is this small subset, not the full HumanML3D/AMASS collection.
A snapshot of the manifest is saved beside the training log.

Task: physics-based reference motion imitation for the 30-DOF custom humanoid.
Rewards track anchor/body/joint pose and body velocities, penalize feet sliding
and undesired contacts. Termination catches excessive anchor and end-effector
errors. Training includes configured randomization/perturbations. This run is
not a language-conditioned policy, and held-out generalization is not established.

After training, the supervisor checks model_9999.pt, then renders each of the
four motion clips for 1000 control steps (~20 seconds), following the robot with
the camera. Videos remain on the remote server under `tw59_s1/videos/`.
Status is `training`, `rendering`, `completed`, or `failed:<exit code>`.
The Linear issue remains In Progress until training and video delivery finish.
