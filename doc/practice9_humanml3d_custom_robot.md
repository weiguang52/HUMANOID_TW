# 实践9：HumanML3D 与自制 30DoF 机器人接入记录

更新时间：2026-08-15

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

当前训练前准备已经完成，可以开始分阶段训练：

- HumanML3D 已完整受控重建并发布 postprocess READY：29,228 个 finite 动作、4,117,392 帧；
- 官方 train/val/test/all split、29,228 份文本、Mean/Std 与全部上游哈希已闭环；
- 前 100 条 bootstrap 重定向有 27 条通过严格质量门；
- 从中按文本筛出 9 条走路、转向和停止动作，共 7,522 帧；
- 9/9 已用新版 30DoF URDF 做 Isaac FK，retarget 与 FK 均为 quality_pass=true；
- 无 unsafe 参数的 2 环境、1 次 PPO 迭代已通过并生成 checkpoint。

当前限制是重定向仍采用“旧 28DoF Pink IK→30DoF 仿射映射”的 bootstrap 路线，并非新版 URDF 直接 IK；训练 URDF 的动力学与限位也是启动参数。它适合仿真课程训练和接口验证，实机部署前仍需新版 URDF 二次 IK 与真实机械参数。

## 数据盘布局

| 内容 | 路径 |
|---|---|
| 仓库 | /root/gpufree-data/projects/HUMANOID_TW |
| 旧 URDF 与 IK_V2 | /root/gpufree-data/projects/IK_V2 |
| 新版原始 URDF | /root/gpufree-data/projects/urdf0711 |
| HumanML3D 官方源码 | /root/gpufree-data/datasets/HumanML3D-official |
| AMASS 官方归档 | /root/gpufree-data/datasets/humanml3d/amass_archives |
| HumanML3D 受控重建 staging | /root/gpufree-data/datasets/practice9/humanml3d_rebuild/staging-v1 |
| HumanML3D READY 动作根 | /root/gpufree-data/datasets/practice9/humanml3d_rebuild/staging-v1/HumanML3D |
| postprocess READY | /root/gpufree-data/datasets/practice9/humanml3d_rebuild/staging-v1/manifests/postprocess-ready.json |
| 训练 URDF | /root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf |
| USD 缓存 | /root/gpufree-data/datasets/practice9/custom_robot/usd |
| 100 条严格质量筛选 | /root/gpufree-data/datasets/practice9/humanml3d_custom30_screen100 |
| 400 条 locomotion 候选 | /root/gpufree-data/datasets/practice9/humanml3d_locomotion_candidates400_v1 |
| 165 条 locomotion FK NPZ | /root/gpufree-data/datasets/practice9/humanml3d_custom30_locomotion400_v1/npz |
| 默认训练 manifest | /root/gpufree-data/datasets/practice9/humanml3d_custom30_locomotion400_v1/manifest.json |
| 训练日志 | /root/gpufree-data/projects/HUMANOID_TW/logs/rsl_rl |
| Isaac 临时日志 | /root/gpufree-data/tmp/practice9_custom |

所有数据、环境、缓存和日志均在 /root/gpufree-data。没有把挂载盘文件复制到本地。AMASS、动作 NPZ、Cookie 和人体模型不进入 Git。

## HumanML3D 与 AMASS 获取

已取得并解压 HumanML3D 官方公开代码、notebook、文本标注、split、HumanAct12 包和 new_joints/012314.npy 校验样本。

18 项 AMASS SMPL+H G 官方归档已全部下载完成：

- ACCAD、BMLhandball、BMLmovi、BMLrub/BioMotionLab_NTroje；
- CMU、DFaust、EKUT、EyesJapanDataset；
- HDM05、HumanEva、KIT、MoSh、PosePrior；
- SFU、SSM、TCDHands、TotalCapture、Transitions。

18 包压缩后合计 10,305,703,324 字节，约 9.60 GiB。全量归档审计结果：bzip2 CRC 18/18 通过，普通文件 14,072 个，流式展开总量 24,634,346,711 字节，没有路径逃逸、链接或设备文件异常。归档没有整体解压到磁盘。

已按各门户独立许可取得 HumanML3D 所需 4 个模型：

- body_models/smplh/{male,female}/model.npz；
- body_models/dmpls/{male,female}/model.npz。

模型、归档、凭据和 Cookie 都只保存在数据盘且不进入 Git。

官方 index.csv 已只读验证：14,616 行、11,715 个唯一 source，其中 HumanAct12 为 1,191 个。

## HumanML3D 受控重建入口

新增 scripts/practice9/rebuild_humanml3d.py，实现 preflight、pose、verify-pose 和 status 四个可恢复 stage。它只读官方目录和归档，输出固定写入 staging；不会覆盖 HumanML3D-official，也不会整体解压 AMASS。

新增 scripts/practice9/rebuild_humanml3d_postprocess.py，实现 segment、mirror、represent、stats、verify 和 ready。后处理持有 pose 共享锁与自身独占锁，使用独立 SQLite ledger、原子文件和逐阶段 marker。

安全边界包括：路径与归档成员类型校验、18 个规范顶层映射、HumanAct12 精确的 humanact12/humanact12 两层布局、输入 SHA/mtime 绑定、原子 NPY/JSON 写入、单实例锁、SQLite 断点状态以及 shape/finite/计数验收。preflight 会流式解压扫描全部 18 包，耗时较长，但不落地归档内容。

纯 CPU 推进顺序：

    cd /root/gpufree-data/projects/HUMANOID_TW
    export CUDA_VISIBLE_DEVICES=""
    export PYTHONDONTWRITEBYTECODE=1
    HML_PY=/root/gpufree-data/conda_envs/humanml3d_cpu/bin/python

    "$HML_PY" -B scripts/practice9/rebuild_humanml3d.py status
    "$HML_PY" -B scripts/practice9/rebuild_humanml3d.py preflight
    "$HML_PY" -B scripts/practice9/rebuild_humanml3d.py pose --device cpu
    "$HML_PY" -B scripts/practice9/rebuild_humanml3d.py verify-pose

BodyModel 的 pose 阶段可在之后使用单张空闲 GPU 加速，但每次启动前必须重新检查 nvidia-smi。脚本要求明确指定物理卡、只暴露一张卡，并在已有计算进程、显存超过 256 MiB 或利用率超过 5% 时拒绝启动：

    nvidia-smi
    CUDA_VISIBLE_DEVICES=1 "$HML_PY" -B \
      scripts/practice9/rebuild_humanml3d.py pose \
      --device cuda --gpu-id 1 --batch-frames 256

本次实际完成结果：

- pose：11,715 个唯一源动作，2,663,038 帧；
- segment+mirror：29,232 个 joints 文件，4,146,624 帧；
- representation：29,228 个 new_joints 和 29,228 个 new_joint_vecs，4,117,392 帧；
- 仅 007975/M007975 使用确定性退化修复，全部输出 finite；
- Mean.npy/Std.npy 按官方 float32 拼接归约生成，Mean 最大参考误差 6.69e-4、Std 最大参考误差 9.15e-5；
- 另保存数值稳定的 Mean_stable/Std_stable，以及包含两个修复动作的 Mean_all_finite/Std_all_finite；
- postprocess verify 与 ready 均已通过。

后处理复现顺序：

    "$HML_PY" -B scripts/practice9/rebuild_humanml3d_postprocess.py segment
    "$HML_PY" -B scripts/practice9/rebuild_humanml3d_postprocess.py represent
    "$HML_PY" -B scripts/practice9/rebuild_humanml3d_postprocess.py stats
    "$HML_PY" -B scripts/practice9/rebuild_humanml3d_postprocess.py verify
    "$HML_PY" -B scripts/practice9/rebuild_humanml3d_postprocess.py ready

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

正式 locomotion400_v1 结果：

| 指标 | 值 |
|---|---:|
| 动作数/总帧数 | 165 / 152,439 |
| retarget quality_pass | 165/165（400 个候选中另有 235 个被拒绝） |
| FK quality_pass | 165/165 |
| 最大 base 位置/姿态回读误差 | 0 / 0 |
| 最低鞋底 | 0.001 m |
| 最大 body 线速度 FD P95 误差 | 0.0436661 m/s |
| 最大 body 角速度 FD P95 误差 | 0.00193807 rad/s |
| manifest SHA256 | ee8f53210109176070249a17bf7f09636c01fa6ec644955dfac98c27adeaabe9 |

公开样本 012314 仍因右肩长期贴限等问题被拒绝，不进入正式训练。converter 使用 SimulationApp 官方 immediate shutdown，单条 FK 回归 12.1 秒自然退出并释放 GPU，避免批处理完成后卡在 Kit 清理。

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

1. HumanML3D pose/postprocess 测试共 43/43 通过；py_compile、bash -n、git diff --check 通过；postprocess 脚本及其测试通过 ruff 检查。
2. 全量 postprocess verify/ready 通过；split 计数为 train 23,384、val 1,460、test 4,384、all 29,228。
3. URDF：30 active + 4 fixed、无 continuous、Pinocchio nq=nv=30。
4. 多动作名称重排、边界、NaN/缺 body 拒绝通过。
5. 50/51/99/100 帧尾 bin 公平性、末帧排除、质量门通过。
6. 旧 G1 6574×29 NPZ 兼容加载通过。
7. 400 条语义平衡 locomotion 候选严格重定向：165 pass、235 fail；总帧 152,439。
8. 新 FK：165/165 pass，30 关节、31 刚体、50 Hz，回读、鞋底和速度有限差分检查通过。
9. 阶段二 model_998 在 165 条 manifest 上续训烟测：2 env、1 PPO iter，171→30、291→1、48 steps、无 NaN/Inf。
10. smoke checkpoint 大小 6,861,099 字节，SHA256 为 2ccd8d555825b21e5cd9f8edd2c68072c0d9f98665f66af40a7197b00de5ab9e。
11. checkpoint：logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d/2026-08-15_21-00-22_practice9_humanml3d_locomotion165_resume_smoke/model_998.pt。
12. GPU 测试后均已释放。
13. Isaac/CUDA/Omniverse/W&B/pip/temp 路径均位于数据盘。

## 文件变更

新增：

- scripts/practice9/prepare_custom_robot_urdf.py
- scripts/practice9/retarget_humanml3d.py
- scripts/practice9/select_humanml3d_locomotion.py
- scripts/practice9/custom_motion_to_npz.py
- scripts/practice9/download_humanml3d_amass.sh
- scripts/practice9/train_humanml3d_custom.sh
- scripts/practice9/rebuild_humanml3d.py
- scripts/practice9/rebuild_humanml3d_postprocess.py
- tests/practice9/test_rebuild_humanml3d.py
- tests/practice9/test_rebuild_humanml3d_postprocess.py
- tests/practice9/test_select_humanml3d_locomotion.py
- source/unitree_rl_lab/unitree_rl_lab/assets/robots/custom_humanoid.py
- source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/motion_library.py
- source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/custom_30dof/humanml3d/

修改 commands.py、observations.py、rewards.py、terminations.py，增加多动作、速度观测、奖励和 motion_end；训练脚本默认使用 locomotion400_v1 的 165 条严格质量 manifest。

## 本次实际构建与扩展方法

生成训练 URDF：

    cd /root/gpufree-data/projects/HUMANOID_TW
    source /opt/conda/etc/profile.d/conda.sh
    conda activate /root/gpufree-data/conda_envs/env_isaaclab
    python scripts/practice9/prepare_custom_robot_urdf.py

当前默认输入已经指向 READY 数据。扩展动作集时先做受控小批量筛选，不要直接把 29,228 条全部放进 eager MotionLibrary：

    python scripts/practice9/select_humanml3d_locomotion.py \
      --new-joints-dir /root/gpufree-data/datasets/practice9/humanml3d_rebuild/staging-v1/HumanML3D/new_joints \
      --texts-dir /root/gpufree-data/datasets/HumanML3D-official/HumanML3D/texts \
      --output-dir /root/gpufree-data/datasets/practice9/humanml3d_locomotion_candidates400_v1 \
      --limit 400 --min-frames 40 --max-frames 300


    python scripts/practice9/retarget_humanml3d.py \
      --input /root/gpufree-data/datasets/practice9/humanml3d_locomotion_candidates400_v1/new_joints \
      --output-dir /root/gpufree-data/datasets/practice9/humanml3d_custom30_locomotion400_v1/retargeted \
      --continue-on-error

正式数据不要加 --allow-quality-failures。先检查 retarget_manifest.json 中 motions 非空、所有 quality_pass=true，并保持总帧数不超过 500,000。

再做 Isaac FK：

    source /root/gpufree-data/conda_envs/env_isaaclab/etc/conda/activate.d/isaac_paths.sh
    export GPUFREE_DATA_ROOT=/root/gpufree-data
    export PRACTICE9_CUSTOM_URDF=/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf
    export PRACTICE9_CUSTOM_USD_DIR=/root/gpufree-data/datasets/practice9/custom_robot/usd
    export CUDA_VISIBLE_DEVICES=1
    export TMPDIR=/root/gpufree-data/tmp/practice9_custom
    export XDG_CACHE_HOME=/root/gpufree-data/.cache/xdg
    export CUDA_CACHE_PATH=/root/gpufree-data/.cache/nvidia/practice9_custom
    export OMNI_USER_DIR=/root/gpufree-data/.cache/omniverse/practice9_custom
    export PYTHONPATH=/root/gpufree-data/projects/HUMANOID_TW/source/unitree_rl_lab:${PYTHONPATH:-}

    python scripts/practice9/custom_motion_to_npz.py \
      --headless --device cuda:0 \
      --input-manifest /root/gpufree-data/datasets/practice9/humanml3d_custom30_locomotion400_v1/retarget_manifest.json \
      --output-dir /root/gpufree-data/datasets/practice9/humanml3d_custom30_locomotion400_v1/npz \
      --output-manifest /root/gpufree-data/datasets/practice9/humanml3d_custom30_locomotion400_v1/manifest.json \
      --continue-on-error # strict batch

运行前必须确认目标 GPU 空闲。正式数据不要加 --allow-quality-failures，也不要设置 P9_CUSTOM_ALLOW_UNSAFE_MOTIONS。

## 如何开始训练

> Current 165-motion stage-3 instructions: [practice9_stage3_locomotion165.md](practice9_stage3_locomotion165.md).
> The locomotion9 commands below are retained as historical stage-1 examples; use the linked document for the current run.

前置条件：

- postprocess-ready.json 存在且 ready=true；
- Default locomotion400_v1 manifest: 165 motions, all quality_pass=true.
- 选择空闲 GPU。

先复现最小烟测：

    cd /root/gpufree-data/projects/HUMANOID_TW
    P9_CUSTOM_GPU=1 \
    P9_CUSTOM_NUM_ENVS=2 \
    P9_CUSTOM_MAX_ITERATIONS=1 \
    P9_CUSTOM_RUN_NAME=practice9_humanml3d_locomotion9_smoke \
    bash scripts/practice9/train_humanml3d_custom.sh

烟测通过后运行 256 环境、200 迭代：

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

1. 先用 locomotion9_v1 完成 200 迭代阶段训练并观察终止率、足滑和跟踪误差。
2. 从 train split 扩展到 50–200 条站立、步行和转向动作，继续严格质量筛选。
3. 在新版 URDF 上增加 Pink/QP 二次 IK，旧 28→30 映射仅作 warm start。
4. 用真实机械与电机参数替换 bootstrap 动力学与单侧膝伸直限位。
5. 名义动力学收敛后逐步增加 push 和 domain randomization。
6. 超过 500,000 帧前实现分片或 CPU cache MotionLibrary。
