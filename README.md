# Unitree RL Lab

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
