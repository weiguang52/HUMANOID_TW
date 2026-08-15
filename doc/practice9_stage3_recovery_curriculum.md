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
