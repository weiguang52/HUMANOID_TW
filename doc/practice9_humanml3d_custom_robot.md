# 实践9：HumanML3D 与自制 30DoF 机器人接入记录

更新时间：2026-08-14

分支：practice9-adaptive-sampling

## 当前结论

实践9已经接通以下代码链路：

    HumanML3D new_joints [T,22,3]，20 Hz
      → IK_V2 旧机器人 28DoF 重定向
      → 关节轴、零位和名称映射到新版机器人 30DoF
      → 恢复根平移和 yaw，统一为 50 Hz
      → 限位、速度、根轨迹质量门
      → Isaac Lab 按新版 URDF 做 FK
      → 带 joint_names/body_names 的多动作 NPZ
      → 30DoF mimic RL

新任务 ID：

    Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D

代码已完成 2 环境、1 次 PPO 迭代冒烟：actor 171→30、critic 291→1。旧 G1 实践9也完成同规模回归，仍为 actor 154→29、critic 286→1。

当前不能直接开始正式训练。RL 链路已运行，但完整 HumanML3D 还缺独立许可的 SMPL+H/DMPL 模型；公开样本 012314 又存在旧 IK 长时间贴限，正式任务会按质量门拒绝它。该样本只用于代码冒烟。

## 数据盘布局

| 内容 | 路径 |
|---|---|
| 仓库 | /root/gpufree-data/projects/HUMANOID_TW |
| 旧 URDF 与 IK_V2 | /root/gpufree-data/projects/IK_V2 |
| 新版原始 URDF | /root/gpufree-data/projects/urdf0711 |
| HumanML3D 官方源码 | /root/gpufree-data/datasets/HumanML3D-official |
| AMASS 官方归档 | /root/gpufree-data/datasets/humanml3d/amass_archives |
| 训练 URDF | /root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf |
| USD 缓存 | /root/gpufree-data/datasets/practice9/custom_robot/usd |
| 重定向输出 | /root/gpufree-data/datasets/practice9/humanml3d_custom30/retargeted |
| Isaac FK NPZ | /root/gpufree-data/datasets/practice9/humanml3d_custom30/npz |
| 最终 manifest | /root/gpufree-data/datasets/practice9/humanml3d_custom30/manifest.json |
| 训练日志 | /root/gpufree-data/projects/HUMANOID_TW/logs/rsl_rl |
| Isaac 临时日志 | /root/gpufree-data/tmp/practice9_custom |

所有数据、环境、缓存和日志均在 /root/gpufree-data。没有把挂载盘文件复制到本地。AMASS、动作 NPZ、Cookie 和人体模型不进入 Git。

## HumanML3D 与 AMASS 获取

已取得并解压 HumanML3D 官方公开代码、notebook、文本标注、split、HumanAct12 包和 new_joints/012314.npy 校验样本。

已按用户授权接受并启动下载 18 项 AMASS SMPL+H G 数据：

- ACCAD、BMLhandball、BMLmovi、BMLrub/BioMotionLab_NTroje；
- CMU、DFaust、EKUT、EyesJapanDataset；
- HDM05、HumanEva、KIT、MoSh、PosePrior；
- SFU、SSM、TCDHands、TotalCapture、Transitions。

18 包合计 10,305,703,324 字节，约 9.60 GiB。官方下载端单连接约 12–15 KB/s；重启连接后一度返回 403，退避后 BMLhandball 与 BMLmovi 已恢复断点续传，ACCAD 仍在自动重试。下载器已支持每文件独立会话、5 分钟退避、断点续传、字节数检查、bzip2 完整性检查和 HTML 响应拦截。

查看下载状态：

    ps -fp "$(cat /root/gpufree-data/tmp/humanml3d/download_master.pid)"
    tail -n 20 /root/gpufree-data/tmp/humanml3d/download_logs/ACCAD.log
    du -sh /root/gpufree-data/datasets/humanml3d/amass_archives

完整 HumanML3D 还需要 MANO 门户的 Extended SMPL+H 和 SMPL 门户的 DMPL。两者是独立许可，AMASS 授权不覆盖；本次没有擅自接受或下载。

## 训练 URDF 修订

scripts/practice9/prepare_custom_robot_urdf.py 不修改原 URDF，只生成数据盘训练副本：

- 32 个 continuous 改为 30 个有限位 revolute；
- left_hand_roll/right_hand_roll 固定，不进入 RL；
- left_foot_roll/right_foot_roll 保留进入 RL，但没有直接重定向目标；
- 删除空 foot_end.STL 对应链接的错误视觉、碰撞和惯性；
- 左右脚碰撞改为由有效 STL 边界生成的 box；
- mesh 路径解析到数据盘；
- 写入 bootstrap position/effort/velocity limit 和 metadata。

当前零姿态对称，初始根高约 0.264 m；converter 与 articulation 两层均关闭自碰撞。限位、力矩、速度、质量和惯量仍是启动参数，实机部署前必须按 CAD、BOM、电机和减速器规格替换。

## 30DoF 控制顺序

    left_hip_linkage_pitch, left_thigh_roll, left_knee_linkage_yaw,
    left_mid_leg_pitch, left_calf_yaw, left_ankle_pitch, left_foot_roll,
    right_hip_linkage_pitch, right_thigh_roll, right_knee_linkage_yaw,
    right_mid_leg_pitch, right_calf_yaw, right_ankle_pitch, right_foot_roll,
    waist_yaw, gearbox_roll, chest_pitch,
    left_shoulder_linkage_pitch, left_upper_arm_roll,
    left_elbow_linkage_pitch, left_force_arm_yaw, left_wrist_pitch,
    right_shoulder_linkage_pitch, right_upper_arm_roll,
    right_elbow_linkage_pitch, right_force_arm_yaw, right_wrist_pitch,
    neck, neck_linkage_roll, head_pitch

动作、观测、重定向、NPZ 和仿真均按名称重排，不依赖 URDF、Pinocchio 或 PhysX 的隐式数组顺序。

## 28→30 启动映射

- 左腿：hip pitch -、hip roll +、hip yaw +、knee -、ankle yaw -、ankle pitch +，foot roll=0。
- 右腿：hip pitch -、hip roll +、hip yaw +、knee +、ankle yaw +、ankle pitch +，foot roll=0。
- 腰：waist yaw +，waist roll→gearbox roll +，waist pitch→chest pitch -。
- 左臂：shoulder pitch -，upper arm roll=q+π/2，shoulder yaw +，elbow +，wrist -。
- 右臂：shoulder pitch -，upper arm roll=q-π/2，shoulder yaw +，elbow +，wrist -。
- 颈：yaw +、roll +、pitch -。

轴符号经过 URDF FK 扰动检查。由于新旧骨长和腰部旋转顺序不同，该仿射映射适合接入、筛选和新版 IK warm start，不代表最终高保真重定向。后续应在新版 30DoF URDF 上增加 Pink/QP 二次优化。

## 根轨迹

HumanML3D new_joints 是 [T,22,3]、20 Hz、Y 向上。旧代码逐帧减 pelvis 并去 yaw，会删除行走和转弯。本次将局部姿态送入 IK，pelvis 平移与髋部朝向单独恢复为 root_pos/root_quat。

坐标保持 IK_V2 约定：

    robot_x = human_z
    robot_y = human_x
    robot_z = human_y

第一帧位置和 yaw 归零，后续位移和转向保留，最终统一到 50 Hz。

## 动作质量门

retarget_humanml3d.py 默认拒绝：

- 任一关节在限位 1% 或 0.001 rad 内的帧比例大于 5%；
- 任一关节连续贴限超过 0.25 s；
- 速度超限样本比例大于 0.5%；
- 任一关节速度比 P99 大于 1.0；
- 任一关节速度比最大值大于 1.2；
- 根水平速度 P99/最大值超过 0.6/0.8 m/s；
- 根 yaw 速度 P99/最大值超过 2/3 rad/s；
- 所需整体时间拉伸超过上限。

非饱和动作会整体时间拉伸并重算根轨迹。位置贴限只能重新 IK 或丢弃。manifest 保存每关节贴限比例、最长时间、速度比和根速度。

训练任务默认要求 quality_pass=true。P9_CUSTOM_ALLOW_UNSAFE_MOTIONS=1 只允许代码冒烟，禁止正式训练使用。

## Isaac FK 与鞋底对地

custom_motion_to_npz.py 会：

- 按 30 个名称写入并回读状态，误差阈值 1e-5；
- 从训练 URDF 读取脚碰撞 box；
- 将 box 的 8 个角点变换到世界系；
- 用真实鞋底最低点对地，默认保留 1 mm；
- 校验 base/root 一致；
- 比较 body velocity 与位姿有限差分；
- 输出 fps、joint_names、body_names、motion_id 和质量字段。

012314 短版结果：

| 指标 | 值 |
|---|---:|
| 帧数 | 423 |
| 关节/刚体 | 30/31 |
| 状态与 base/root 回读误差 | 0 |
| 对地平移 | -0.002074 m |
| 最低鞋底 | 0.001 m |
| body 线速度 FD P95 误差 | 0.0691 m/s |
| body 角速度 FD P95 误差 | 0.0269 rad/s |
| quality_pass | false |

开启时间拉伸需要约 5.60 倍并扩到 2364 帧，但不能修复右肩位置饱和，因此它不会进入正式训练。

## 多动作与 RL

MotionLibrary 支持单 NPZ 和 schema v1 manifest：

- 每环境独立 motion_id/local_frame；
- bin 不跨 clip；
- 最后一帧只作正常 time_out，不作随机起点；
- 尾部短 bin 按可起始帧数加权；
- failure 与 uniform prior 分开；
- 动作自然结束不会在 PPO 中无标记跳变；
- 校验 FPS、形状、名称、NaN/Inf 和四元数；
- eager 后端默认最多 500,000 帧。

首期只使用经过质量筛选的几十到几百条站立、步行和转向。全量需要后续分片/mmap/cache 后端。

新任务 action=30，actor observation=171，critic observation=291。奖励包含 anchor、core/legs/arms、body pose/velocity、28 关节软跟踪、feet slide、接触和控制正则。两个 foot_roll 可控但不直接跟踪，hand_roll 固定。初训关闭 push，随机化较小。

## 验证结果

1. compileall、bash -n、git diff --check 通过。
2. URDF：30 active + 4 fixed、无 continuous、Pinocchio nq=nv=30。
3. 多动作名称重排、边界、NaN/缺 body 拒绝通过。
4. 50/51/99/100 帧尾 bin 公平性、末帧排除、质量门通过。
5. 旧 G1 6574×29 NPZ 兼容加载通过。
6. 新 FK：30 关节、31 刚体、50 Hz，回读和鞋底检查通过。
7. 新任务 2 env、1 PPO iter：171→30、291→1、48 steps、无 NaN/Inf。
8. 旧任务回归：154→29、286→1、48 steps。
9. GPU 测试后均已释放。
10. Isaac/CUDA/Omniverse/W&B/pip/temp 路径均位于数据盘。

## 文件变更

新增：

- scripts/practice9/prepare_custom_robot_urdf.py
- scripts/practice9/retarget_humanml3d.py
- scripts/practice9/custom_motion_to_npz.py
- scripts/practice9/download_humanml3d_amass.sh
- scripts/practice9/train_humanml3d_custom.sh
- source/unitree_rl_lab/unitree_rl_lab/assets/robots/custom_humanoid.py
- source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/motion_library.py
- source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/custom_30dof/humanml3d/

修改 commands.py、observations.py、rewards.py、terminations.py，增加多动作、速度观测、奖励和 motion_end。

## 数据到齐后的构建

生成训练 URDF：

    cd /root/gpufree-data/projects/HUMANOID_TW
    source /opt/conda/etc/profile.d/conda.sh
    conda activate /root/gpufree-data/conda_envs/env_isaaclab
    python scripts/practice9/prepare_custom_robot_urdf.py

官方流程生成完整 new_joints 后，先做 200 条小课程：

    python scripts/practice9/retarget_humanml3d.py \
      --input /root/gpufree-data/datasets/HumanML3D-official/HumanML3D/new_joints \
      --limit 200 \
      --continue-on-error

再做 Isaac FK：

    export GPUFREE_DATA_ROOT=/root/gpufree-data
    export PRACTICE9_CUSTOM_URDF=/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf
    export PRACTICE9_CUSTOM_USD_DIR=/root/gpufree-data/datasets/practice9/custom_robot/usd
    export CUDA_VISIBLE_DEVICES=1
    export TMPDIR=/root/gpufree-data/tmp/practice9_custom
    export XDG_CACHE_HOME=/root/gpufree-data/.cache/xdg
    export CUDA_CACHE_PATH=/root/gpufree-data/.cache/nvidia/practice9_custom
    export OMNI_USER_DIR=/root/gpufree-data/.cache/omniverse/practice9_custom
    export PYTHONPATH=/root/gpufree-data/projects/HUMANOID_TW/source/unitree_rl_lab
    export LD_LIBRARY_PATH=/root/gpufree-data/isaacsim

    python scripts/practice9/custom_motion_to_npz.py \
      --headless --device cuda:0 \
      --input-manifest /root/gpufree-data/datasets/practice9/humanml3d_custom30/retarget_manifest.json \
      --output-dir /root/gpufree-data/datasets/practice9/humanml3d_custom30/npz \
      --output-manifest /root/gpufree-data/datasets/practice9/humanml3d_custom30/manifest.json

正式数据不要加 --allow-quality-failures。

## 如何开始训练

前置条件：

- AMASS 完整下载并校验；
- 合法取得 SMPL+H/DMPL，生成完整 new_joints；
- 最终 manifest 至少有一条 quality_pass=true；
- 选择空闲 GPU。

先运行 256 环境、200 迭代：

    cd /root/gpufree-data/projects/HUMANOID_TW
    P9_CUSTOM_GPU=1 \
    P9_CUSTOM_NUM_ENVS=256 \
    P9_CUSTOM_MAX_ITERATIONS=200 \
    P9_CUSTOM_RUN_NAME=practice9_humanml3d_custom30_stage1 \
    bash scripts/practice9/train_humanml3d_custom.sh

确认无 NaN、奖励和终止率正常后，再扩大：

    P9_CUSTOM_GPU=1 \
    P9_CUSTOM_NUM_ENVS=1024 \
    P9_CUSTOM_MAX_ITERATIONS=10000 \
    P9_CUSTOM_RUN_NAME=practice9_humanml3d_custom30_full \
    bash scripts/practice9/train_humanml3d_custom.sh

启动脚本在物理 GPU 显存超过 256 MiB 时拒绝运行，避免阻碍其他训练。正式训练不要设置 P9_CUSTOM_ALLOW_UNSAFE_MOTIONS。

## 后续建议

1. 取得 SMPL+H/DMPL 独立许可并完成官方重建。
2. 先筛选站立、步行和转向。
3. 在新版 URDF 上增加 Pink/QP 二次 IK，旧映射仅作 warm start。
4. 用真实机械与电机参数替换 bootstrap 动力学。
5. 名义动力学收敛后逐步增加 push 和 domain randomization。
6. 超过 500,000 帧前实现分片或 CPU cache MotionLibrary。
