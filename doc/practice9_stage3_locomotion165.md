# 实践9：165条动作阶段三训练

> 该阶段已完成但未通过验收，不能继续执行本文末尾的1024环境命令。退化分析和修正版
> 课程训练请见 `doc/practice9_stage3_recovery_curriculum.md`。

## 当前状态

- HumanML3D locomotion 候选：400条，按前进、转向、后退、侧移、原地动作平衡选择。
- 严格重定向：165条通过，235条因关节限位或动态质量门被拒绝。
- Isaac FK：165/165通过，共152,439帧，30关节、31刚体、50 Hz。
- 正式 manifest：`/root/gpufree-data/datasets/practice9/humanml3d_custom30_locomotion400_v1/manifest.json`。
- manifest SHA256：`ee8f53210109176070249a17bf7f09636c01fa6ec644955dfac98c27adeaabe9`。
- eager MotionLibrary 上限为500,000帧；当前数据量在限制内。
- 阶段二 checkpoint：`2026-08-15_17-16-50_practice9_humanml3d_custom30_stage2/model_998.pt`。
- 165条数据续训烟测已通过：2 env、1 iteration，actor 171→30，critic 291→1。

正式数据没有使用 `--allow-quality-failures`，训练时不要设置 `P9_CUSTOM_ALLOW_UNSAFE_MOTIONS`。

## 阶段三命令

先确认 GPU 1 空闲：

```bash
nvidia-smi -i 1
```

然后从阶段二 `model_998.pt` 在165条动作上继续训练：

```bash
cd /root/gpufree-data/projects/HUMANOID_TW

P9_CUSTOM_GPU=1 \
P9_CUSTOM_NUM_ENVS=512 \
P9_CUSTOM_MAX_ITERATIONS=1000 \
P9_CUSTOM_RUN_NAME=practice9_humanml3d_custom30_stage3_locomotion165 \
bash scripts/practice9/train_humanml3d_custom.sh \
  --resume \
  --load_run 2026-08-15_17-16-50_practice9_humanml3d_custom30_stage2 \
  --checkpoint model_998.pt
```

训练脚本默认已经指向165条 manifest；也可显式设置：

```bash
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/humanml3d_custom30_locomotion400_v1/manifest.json
```

## 阶段三验收重点

不要只看总 reward，至少同时检查：

1. 日志和 checkpoint 参数中没有 NaN/Inf。
2. `Episode_Termination/motion_end` 随训练上升。
3. `anchor_ori`、`anchor_pos`、`ee_body_pos` 终止率没有持续恶化。
4. `Metrics/motion/error_body_pos` 和 `error_joint_pos` 总体下降。
5. `sampling_entropy` 没有快速塌缩，`sampling_top1_prob` 没有长期集中于少数动作。
6. 播放抽检前进、转弯、后退、侧移和原地动作，而不是只看一个片段。

## 下一阶段

阶段三稳定后，从其最新 checkpoint 继续扩大到1024环境；将占位符替换为实际阶段三运行目录和checkpoint：

```bash
cd /root/gpufree-data/projects/HUMANOID_TW

P9_CUSTOM_GPU=1 \
P9_CUSTOM_NUM_ENVS=1024 \
P9_CUSTOM_MAX_ITERATIONS=5000 \
P9_CUSTOM_RUN_NAME=practice9_humanml3d_custom30_full_locomotion165 \
bash scripts/practice9/train_humanml3d_custom.sh \
  --resume \
  --load_run <stage3-run-directory> \
  --checkpoint <latest-model.pt>
```

在165条名义动力学训练稳定前，不要提前增大 push 或 domain randomization。实机部署前还必须把 bootstrap 限位、惯量、执行器刚度/阻尼替换为真实参数，并将旧28DoF到30DoF仿射映射升级为新版URDF直接IK或二次优化。
