# 实践9：阶段三退化修复与渐进课程

## 结论

原阶段三运行 `2026-08-15_21-57-55_practice9_humanml3d_custom30_stage3_locomotion165`
完整跑到 `model_1997.pt`，checkpoint 全部张量 finite，但没有通过训练验收：后100次迭代平均
reward约1.52、平均回合约25.8步、动作自然完成率约9.2%。采样熵降到约0.39，单个bin
采样概率升到约13.2%，说明失败采样形成了正反馈。该checkpoint仅保留作诊断，不作为后续训练起点。

## 修改内容

- 失败难度由原始失败次数改为每个bin的EMA失败率，即失败次数除以访问次数。
- 未访问bin初始难度设为1，避免尚未探索的动作被饿死。
- `adaptive_uniform_ratio`现在表示明确的先验混合质量，不再是加到原始计数上的微小数值。
- 自制机器人任务使用70%均匀先验、30%困难采样，并限制任一bin最大概率为2%。
- 新增归一化有效bin数、失败率均值和失败率最大值监控。
- 从阶段二9条稳定动作出发，构建40、80、165、174条嵌套课程；只写manifest，不复制NPZ。

课程目录：

`/root/gpufree-data/datasets/practice9/humanml3d_custom30_curriculum_v1`

| 阶段 | 动作数 | 帧数 | 说明 |
|---|---:|---:|---|
| `stage_040.json` | 40 | 41,996 | 9条阶段二种子 + 31条低难度平衡动作 |
| `stage_080.json` | 80 | 78,467 | 在40条基础上扩展 |
| `stage_165.json` | 165 | 151,564 | 9条种子 + 原165条中较容易的156条 |
| `stage_174.json` | 174 | 159,961 | 阶段二9条种子 + 原165条全部动作 |

四个阶段均低于MotionLibrary的500,000帧上限，所有动作保持原严格质量门结果。

## 运行时smoke结果

修正版已从阶段二 `model_998.pt` 在 `stage_040.json` 上完成64环境、20迭代smoke：

- 最终checkpoint：`2026-08-16_01-02-29_practice9_sampler_curriculum40_smoke64/model_1017.pt`。
- actor、critic和optimizer全部张量finite，维度保持actor 171到30、critic 291到1。
- `sampling_top1_prob`最大0.00120，显著低于0.02上限。
- 最后一次 `sampling_entropy` 为0.998，`sampling_effective_bins` 为0.981。
- 最后一次平均reward为9.10、平均回合长度152.9步、`motion_end`为0.334。
- GPU在退出后释放，无遗留训练进程。

短smoke只验证实现闭环，不能替代下面500迭代的正式阶段验收。

## 第一阶段重训命令

必须从阶段二稳定的 `model_998.pt` 重启，不要从退化的 `model_1997.pt` 续训：

```bash
cd /root/gpufree-data/projects/HUMANOID_TW

PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/humanml3d_custom30_curriculum_v1/stage_040.json \
P9_CUSTOM_GPU=1 \
P9_CUSTOM_NUM_ENVS=512 \
P9_CUSTOM_MAX_ITERATIONS=500 \
P9_CUSTOM_RUN_NAME=practice9_stage3a_balanced40 \
bash scripts/practice9/train_humanml3d_custom.sh \
  --resume \
  --load_run 2026-08-15_17-16-50_practice9_humanml3d_custom30_stage2 \
  --checkpoint model_998.pt
```

## 晋级门槛

每一级至少检查最后100次迭代：

1. `sampling_top1_prob`不得超过0.02，`sampling_entropy`不应持续下降。
2. `sampling_effective_bins`应保持稳定或上升。
3. `motion_end`应总体上升；40条阶段建议达到0.35以上再扩展。
4. `anchor_pos`和`ee_body_pos`终止率不得连续恶化。
5. `error_body_pos`和线速度误差不得在后半程持续上升。
6. checkpoint所有actor、critic和optimizer张量必须finite。

40条通过后，从其最新checkpoint依次切换到 `stage_080.json`、`stage_165.json` 和
`stage_174.json`。每次只扩大一个阶段，先做2环境smoke，再做正式训练。165或174条稳定前，
不要扩大到1024环境，也不要增强push或domain randomization。

## 174动作巩固与逐动作评估

1024环境巩固运行：

`2026-08-16_03-08-53_practice9_stage4_consolidate174`

从 `model_3000.pt` 在完整174动作上继续600次迭代。最后100次平均reward为39.16，动作自然完成率
为83.31%，anchor位置、姿态和末端终止率分别为3.02%、6.26%和10.08%；最后50次仍在改善。
最终采用 `model_3599.pt`，actor/critic/optimizer共68个张量全部finite。

为避免训练总体均值掩盖困难动作，新增 `scripts/rsl_rl/evaluate_humanml3d_custom.py`。评估入口通过
`MotionCommand.set_evaluation_motion_ids()`让每个环境永久绑定一个clip，并在每次reset从第0帧开始；成功严格
定义为触发 `motion_end` 且没有失败终止。结果原子发布为 `results.json`、`per_motion.csv` 和
`summary.txt`，同时记录manifest与checkpoint SHA256。

正式评估使用174个并行环境、每条动作3回合、seed 42：

```bash
python scripts/rsl_rl/evaluate_humanml3d_custom.py \
  --headless --device cuda:0 \
  --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D \
  --motion_manifest /root/gpufree-data/datasets/practice9/humanml3d_custom30_curriculum_v1/stage_174.json \
  --episodes_per_motion 3 \
  --checkpoint /root/gpufree-data/projects/HUMANOID_TW/logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d/2026-08-16_03-08-53_practice9_stage4_consolidate174/model_3599.pt \
  --output_dir /root/gpufree-data/datasets/practice9/evaluations/model3599_stage174_eval3_seed42 \
  --seed 42
```

评估结果：522回合中479回合自然完成，成功率91.76%，平均完成比例94.48%；153条动作3/3全过，
10条2/3通过，11条0/3。失败终止以 `ee_body_pos` 为主（36次），anchor位置和姿态各6次。
11条0/3动作是：`000930`、`001882`、`002585`、`005890`、`006564`、`008186`、`009223`、
`011735`、`012605`、`012634`、`013604`。其中10条是forward，说明当前弱点集中在部分前向动作，
不是全局采样坍塌。

## 评估驱动的困难动作微调

`scripts/practice9/build_evaluation_curriculum.py`将完整评估结果转换为加权manifest，不复制NPZ：
3/3动作权重1、2/3动作权重2、0/3动作权重4。生成文件为：

`/root/gpufree-data/datasets/practice9/humanml3d_custom30_eval_hard174_v1/manifest.json`

该文件保留全部174条和159,961帧，并绑定源manifest、评估结果与checkpoint SHA256。2环境、1迭代smoke
`2026-08-16_04-01-26_practice9_evalhard174_smoke2`已通过，checkpoint共68个张量全部finite。
下一轮仍不增加push或domain randomization，从稳定的 `model_3599.pt` 只针对困难动作分布微调：

```bash
cd /root/gpufree-data/projects/HUMANOID_TW

PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/humanml3d_custom30_eval_hard174_v1/manifest.json \
P9_CUSTOM_GPU=1 \
P9_CUSTOM_NUM_ENVS=1024 \
P9_CUSTOM_MAX_ITERATIONS=600 \
P9_CUSTOM_RUN_NAME=practice9_stage4b_evalhard174 \
bash scripts/practice9/train_humanml3d_custom.sh \
  --resume \
  --load_run 2026-08-16_03-08-53_practice9_stage4_consolidate174 \
  --checkpoint model_3599.pt
```

微调完成后必须用同一个逐动作评估入口重新跑3回合对照；只有0/3动作显著减少且原153条没有退化，
才进入摩擦、质量、COM与push的鲁棒性课程。
