# Unitree RL Lab

## TW research and training status

This branch contains two separate research lines. Their results must not be combined into a policy acceptance claim.

| Line | Current status | Evidence |
| --- | --- | --- |
| SuperMimic world-model / imitation | Stopped after failed acceptance; previous policy pilots passed 0/7 fixed independent motions. No accepted TW SuperMimic policy. | Historical handoff below and `docs/practice9/supermimic_dense_static_final_review.json` |
| Isaac Lab stage1 velocity walking | Issue20 complete: 30 formal evaluations, zero training; frozen outcome simulator_not_equivariant. Required symmetric-USD prerequisite stop; original USD skipped. No default promotion; issue17/19 acceptance remains FAILED. | [Issue20 results](docs/stage1_issue20_20261114.md), [archive](validation_artifacts/stage1_issue20_20261114/results.md) |

<!-- ISSUE20_STATUS_START -->
Issue20 complete: 30 formal evaluations, zero training; frozen outcome simulator_not_equivariant. Required symmetric-USD prerequisite stop; original USD skipped. No default promotion; issue17/19 acceptance remains FAILED. The same five final symmetric policies are tested with start phases0/10 and original/strict symmetric USD. Rewards/defaults unchanged. [Full report](docs/stage1_issue20_20261114.md) includes all cell/seed intervals, paired S/D, startup and late slope, half cycles, model differences and colored paired-overhead/front/side videos. Historical findings below remain preserved.
<!-- ISSUE20_STATUS_END -->

Issue #19: all ten fixed-budget continuations and sixty formal evaluations are complete. Symmetric actor forward0.5 heading improves substantially versus matched continuation, but averages48.044deg (threshold30deg), while bounded0.3 fall upper bound1.3802% exceeds1%. Tracking recovery passes; overall acceptance fails. No default promotion. The sole unimplemented next recommendation is the original/reflected policy x original/strict symmetric USD 2x2 diagnostic. See [full report](docs/stage1_issue19_20261113.md) for all commands, seed intervals, exact-state videos and separately accounted user-authorized infrastructure retry. Historical findings below remain preserved.

Issue #18 evaluates all15existing final checkpoints with a structurally equivariant inference wrapper (same102400fixedstates/model: maxE=0; identity bitwise equal). All45new strict jobs and45reused originals are complete:540command records/552960initial trials. sym_int forward0.5absolute heading drops277.231to27.227deg (90.18%; paired95% difference[-328.221,-158.710]), but only one cohort is interpretable and only2of4zero-yaw commands meet30deg. The two control cohorts incur forward0.8fall gains40.443/40.000percentage points and are excluded from causal interpretation. OverallC; no claim of physical asymmetry or accepted straight walking. Full tracking and speed tradeoffs remain reported. See [design and results](docs/stage1_issue18_20261113.md) and [lossless archive](validation_artifacts/stage1_issue18_20261113/results.md), including colored exact-state front/overhead videos with worldXYtrails. Zero training, no default promotion; the single unimplemented next recommendation is the original/reflected-policy x original/strictly-symmetric-USD2x2diagnostic.

Issue #17 tests integer-clock control versus spatial-mirror plus half-cycle data augmentation across five matched seeds per group (15000updates total). All90formal evaluations completed. The primary heading difference is -78.856deg with95% interval[-195.795,+35.015], so improvement is not established. Fixed-state action equivariance improves, but forward0.5XY tracking worsens21.50%, turn tracking worsens135.65%/143.11%, and multiple heading/robustness guards fail. Stop this hypothesis without extra training, checkpoint scan or default promotion. See [design and results](docs/stage1_issue17_20261113.md); the [full results archive](validation_artifacts/stage1_issue17_20261113/results.md) contains lossless raw data, ten final model/optimizer states and colored front/overhead videos.

Stage1 FT5 final checkpoint is `model_5997.pt`. Its archived training metrics remain distinct from the new deterministic evaluation. The full [experiment results](validation_artifacts/stage1_issues_20261108/README.md) include 10 final checkpoints, matched multi-seed comparisons, 32 colored front/side videos and integrity evidence. FT5 passes the sampled original-range fall-rate gate but tracks yaw poorly. The final robust08 policy improves wider-range nominal tracking; material challenge failures remain, so experimental profiles are not promoted to defaults. Stage1 is a separate continuation of robot control work; it does not reopen or validate the stopped SuperMimic experiments. SuperMimic remains stopped, with its unmet acceptance preserved.

Issue #12 uses five matched training-seed pairs, each continued from its own checkpoint after 4,500 requested updates for exactly 1,500 more updates, plus 92,160 initial evaluation trials. At 0.3 m/s, control 140/15,360 = 0.9115% vs treatment 95/15,360 = 0.6185%; the paired treatment-minus-control 95% interval is [-0.4232, -0.1432] percentage points. However, 0.8 m/s xy tracking error increases 22.77%, above the frozen 10% bound. The scheme is rejected overall, with no extra budget, weight scan or default promotion. Full models, contribution monitoring, same-#11 strata and front/side video evidence are in the linked report.

Issue #13 reuses all ten final checkpoints with zero training or configuration changes. All 60 formal evaluation jobs (245,760 initial trials) completed. Under clean conditions at 0.8 m/s, xy error increases 41.15%; combined outside-window error increases 24.55%. Both paired training-seed difference intervals exclude zero, so the frozen diagnosis is B (global side effect); both prespecified sampling sensitivities agree. Clean mean forward speed falls from 0.729 to 0.690 m/s while temporal speed SD decreases, supporting systematic slowing. Five paired training seeds and unadjusted bootstrap 95% intervals limit generalization. The proposed critic-only push-clock hypothesis was subsequently tested in issue #14 below. Raw time decomposition, all intervals, and colored exact-state front/side videos are in the issue #13 report.

Issue #14 adds only two recovery-clock inputs to the critic (134 to 136); the deployed actor stays 101-wide and the switch defaults off. Five matched 4096-environment continuations completed exactly 1500 updates each; 30 new formal evaluation jobs completed, with the original two groups reused unchanged. Clean 0.8 m/s xy error changes from 0.118208 to 0.108004 m/s versus the old treatment, but the paired difference 95% interval [-0.027301, +0.006894] crosses zero, so the frozen primary fails. Tracking versus the old control exceeds the 5% bounds at both 0.5 and 0.8 m/s; clean 0.8 m/s mean speed is still lower, with paired difference -0.023917 m/s [ -0.046628, -0.001206 ]. The two fall-rate gates pass, but full acceptance fails. Stop this hypothesis without added training, representation changes, weight search or default promotion. The linked report includes five full new checkpoints/optimizer states, all three groups, paired intervals, actual-speed/fall caveats, six colored exact-state videos and complete integrity evidence.

Issue #15 evaluates 37 lineage entries (35 distinct checkpoints) under one clean evaluator, plus the three original FT5-protocol reruns: all 108 jobs and 647,424 initial trials completed, with zero training updates. All 225 previous control/treatment/clock command records reproduce exactly, including bitwise original moment arrays. Five multi-seed positive-interval jumps and two single-seed descriptive jumps meet the frozen >20% diagnostic rule; these are unadjusted comparisons with ancestry and fixed-baseline limits, not seven causal discoveries. Body-z yaw variation dominates all 33 straight node/speed cells, with 2.5 Hz peaks at 0.3 m/s matching the 0.4 s gait phase, while large world-heading drift coexists. At 0.3 m/s the control/treatment mean absolute 20 s heading drift is 107.1/238.5 degrees. FT5-to-robust05 body-z yaw MAE increases while heading drift improves, so yaw MAE cannot substitute for actual heading. The report includes complete six-command tables, reset-aware heading, spectra, actual resume/configuration provenance, six figures, four colored front/side videos, raw trial data and integrity evidence. Its sole future recommendation is a separately frozen straight-heading acceptance gate; no reward/configuration change is implemented, and failed issue #12/#14 hypotheses remain stopped.

This distinction implements GitHub issues #1–4, created together on 2026-11-08 at 06:54 UTC. Current-run results are tracked in [Notion](https://www.notion.so/403325d8231d8172995bf19510b8d760). Per the user decision on 2026-11-08, Notion replaces Linear for future task/result tracking; GitHub issue replies remain aligned.

<!-- TW_SUPERMIMIC_HANDOFF_START -->
## TW SuperMimic adaptation handoff

Branch: `supermimic/tw-adaptation-20261104`. This section describes the TW robot; the upstream Unitree instructions below do not establish TW policy acceptance.

**No accepted TW SuperMimic control policy exists.** Previous policy pilots passed 0/7 fixed independent motions. Lower world-model error does not demonstrate stable standing, walking or whole-body imitation.

Implemented: TW30 state/action mapping and nominal shoulders, sole geometry, world-frame angular velocity writes, pose/velocity integration, XY/yaw structure contracts, native replay and paired offline calibration/audits. Total URDF mass is approximately0.841 kg; no hardware mass reference exists.

Latest completed `static_response_common_pair_v1`: static validation H32 root-position trajectory-median P95 improved44.07 to31.97 mm; dynamic139.02 to110.47 mm. Original static limit31.33 mm still fails; original static6/18 metrics and paired branch8/9 checks fail. No promotion. Short-step pose/angular/joint errors, accumulated drift and branch response generalization remain unresolved.

### Final current run: training and primary evaluation completed, rejected

`dense_static_pair_v1`: resume final common candidate/full Adam3000; fixed1000 updates per arm. Retain all old losses; only added dense static H1 supervision coefficient0/1 differs. Both2208 predictions/update, including32 original-static fit points from3305 transitions. Smoke4 control/candidate/repeat-control passes; repeated control model/full Adam bitexact.

Both arms completed1000 updates. Full integrity passes22 frozen tensors,8 Adam states at4000, identical paired samplers and2208 forward points/update. No historical1000 control repeat is available; smoke4 repeat is bitexact.

| Check | Failed / total |
| --- | --- |
| Paired static validation | 0/18 |
| Paired dynamic validation | 8/18 |
| Expanded fit diagnostic (not generalization) | 11/18 |
| Original-reference static validation | 7/18 |
| Original-reference dynamic validation | 0/18 |
| Held branch checks | 7/9 |

Both declared coordinate audits pass. Static H32 root-position trajectory-median P95: control56.46 vs candidate46.58 mm; original reference29.84 mm. Dynamic:137.76 vs133.39 mm. Despite these improvements, the complete gate fails. Candidate is an offline dynamics model, not an accepted policy; no new stable-control effect or policy video is claimed.

Raw final hashes: control `31cea15a7072baffe7471b6d48c1b214e7c9958d2d728c7f0c06f9ade0891d83`; candidate `fed6e7365301e3a0502b4c20d2f26cfad5a4742e5d678adddb8abc9b9965c946`.

User requests stopping the long-term goal after this run and evaluation; no new experiment will be started. All scheduled supplementary audits completed on 2026-11-06; report/model hashes and frozen-state checks verified. No training or audit tmux sessions remain. The long-term acceptance remains unmet; stop follow-up as requested after code handoff. See [final review evidence](docs/practice9/supermimic_dense_static_final_review.json).

### Environment and evidence

All data/cache/artifacts stay on remote data disk `/root/gpufree-data`; preserve Isaac environment. Repo: `projects/HUMANOID_TW`; Python environment: `conda_envs/env_supermimic`.

Experiment root under data disk: `datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/dense_static_pair_v1`. Tmux: `dense_static_pair`, `dense_static_supp`.

Runner: `scripts/practice9/run_supermimic_dense_static_pair.sh`. Existing-output guards prevent overwrite; use new output paths for reproduction, do not restart this run.

Evidence: `smoke4/fit_integrity.json`, final `fit_integrity.json`, `paired_metric_review.json`, `declared/branch_validation.json`, `static_endpoint_bias/audit.json`, `static_rollout_exposure/audit.json`. Missing reports are not passes.

[Detailed calibration history](docs/practice9/supermimic_longterm_calibration_20261031.md). Final policy acceptance remains fixed7 source-isolated motions, full original durations, no original terminations, at least3 independent seeds, same-condition baseline and front/side videos. These requirements remain unmet.
### Git handoff status

Push verified on 2026-11-06 20:31 Asia/Shanghai through a temporary SSH reverse tunnel to the local SOCKS proxy. GitHub branch matched commit709bc82aa661b6ec82ad957637abefd39bd9a7dc before this documentation update. No data-disk files were copied locally. Temporary networking is closed after verification; no persistent proxy setting is required.

<!-- TW_SUPERMIMIC_HANDOFF_END -->


[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/ZwcVwxv5rq)


## Overview

This project provides a set of reinforcement learning environments for Unitree robots, built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab).

Currently supports Unitree **Go2**, **H1** and **G1-29dof** robots.

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="https://oss-global-cdn.unitree.com/static/d879adac250648c587d3681e90658b49_480x397.gif" width="240px">](g1_sim.gif) | [<img src="https://oss-global-cdn.unitree.com/static/3c88e045ab124c3ab9c761a99cb5e71f_480x397.gif" width="240px">](g1_mujoco.gif) | [<img src="https://oss-global-cdn.unitree.com/static/6c17c6cf52ec4e26bbfab1fbf591adb2_480x270.gif" width="240px">](g1_real.gif) |

</div>

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install the Unitree RL IsaacLab standalone environments.

  - Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    git clone https://github.com/unitreerobotics/unitree_rl_lab.git
    ```
  - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    conda activate env_isaaclab
    ./unitree_rl_lab.sh -i
    # restart your shell to activate the environment changes.
    ```
- Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files



- Verify that the environments are correctly installed by:

  - Listing the available tasks:

    ```bash
    ./unitree_rl_lab.sh -l # This is a faster version than isaaclab
    ```
  - Running a task:

    ```bash
    ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
    ```
  - Inference with a trained agent:

    ```bash
    ./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
    # same as
    python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
    ```

## Deploy

After the model training is completed, we need to perform sim2sim on the trained strategy in Mujoco to test the performance of the model.
Then deploy sim2real.

### Setup

```bash
# Install dependencies
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
# Install unitree_sdk2
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install
# Compile the robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make
```

### Sim2Sim

Installing the [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation).

- Set the `robot` at `/simulate/config.yaml` to g1
- Set `domain_id` to 0
- Set `enable_elastic_hand` to 1
- Set `use_joystck` to 1.

```bash
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

### Sim2Real

You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

```bash
./g1_ctrl --network eth0 # eth0 is the network interface name.
```

## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [robot_lab](https://github.com/fan-ziqi/robot_lab): Referenced for project structure and parts of the implementation.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.


## Custom humanoid native motion pipeline

See [TW-56 setup and commands](doc/tw56_native_retarget_pipeline.md) for the active native C++ retargeting, aligned 30-DoF robot, validated Isaac FK and PPO pipeline.

Issue #16 freezes a mean absolute reset-aware20s heading gate of30 degrees for stand/0.3/0.5/0.8 m/s and reuses issue #15 lineage data: none of11 nodes passes all four commands. All34 new formal evaluation jobs (199 command records;203,776 initial trials) completed without training. All five control6000 seeds reverse direction under mirrored inference at every zero-yaw command; requested4500 reverses all five seeds at stand/0.5/0.8 m/s, while0.3 m/s is mixed. This demonstrates a policy-dependent component without establishing exclusive policy causation. The actual USD has measured structural differences, but effective PD/limits/defaults match and no-policy default-PD standing has1024/1024 survivors with only0.000639deg mean absolute drift. Parent-model mirror sensitivity, shared ancestry and censoring prevent a simple inherited-bias-only explanation. The sole unimplemented next proposal is a no-training2x2 original/reflected-policy versus original/strictly symmetric-USD diagnostic. See the [full issue #16 report](validation_artifacts/stage1_issue16_20261111/results.md) for paired seed intervals, all raw trials, physical/runtime audits, seven figures and three colored exact-state views of the original median-heading trial. No reward, production configuration, model or default promotion changes are made.
