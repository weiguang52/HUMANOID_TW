# Issue #5: robust08 failure attribution and training-seed replication

Baseline commit: 3932d7ff380f96293e67aeedb8ab0338f1dd6ae6. Scope is GitHub issue #5, created 2026-11-09 01:00:29 Asia/Shanghai. All artifacts remain on the remote data disk. Defaults are unchanged.

## Paired single-factor evaluation

Existing robust08 seed1 model_8995: clean, push only, friction/mass only, fixed 20 ms delay only, observation noise only, and original combined challenge. Three evaluation seeds, 128 environments, 20 seconds at each of 0.3/0.5/0.8 m/s. Each command explicitly reseeds the full reset with seed*1000+command_index. Non-target startup randomization, pushes, action latency and observation corruption are disabled. Combined retains original random 0/20 ms delay; isolated delay is exactly 20 ms as requested. Numerical criteria compare paired resets within this new protocol, rather than treating historical random streams as identical.

## Frozen selection rule (before single-factor results)

Compute each condition fall rate pooled equally across three commands and three evaluation seeds. A factor is called dominant only if: its excess above clean is at least five percentage points; it exceeds the runner-up by at least five percentage points; its excess is at least half the combined excess; and it ranks first in each evaluation seed after averaging the three commands. Otherwise report no single dominant factor. In that case the highest-scoring isolated factor is only an exploratory stressor hypothesis for the requested training ablations, not an established cause. Ties use push, material, delay, noise order. No thresholds are tuned after observing factor outcomes.

## Training budget and origin

Original robust08 resumed robust05 seed1 model_7496. All five new formal runs use that identical checkpoint and full optimizer: one original robust08 replication with training seed2, plus remove-factor and strengthen-factor arms at training seeds1 and2. Each run gets 4096 environments and1500 PPO updates, no std reset or checkpoint cherry-picking. Variance estimates are conditional on the shared seed1 robust05 parent, not independent replications of the entire historical curriculum.

Remove arm disables only the selected component; all remaining original robust08 settings are retained. Strengthen arm changes only the selected component relative to the original robust08 package: push +/-0.45 m/s; or friction static .3–1.6/dynamic .2–1.3 and base mass addition -.10–.20 kg; or uniformly 0/20/40 ms latency; or double existing observation-noise amplitude. The actual selected modification is frozen in a generated plan after attribution. Non-selected components are unchanged, which is the intended meaning of only strengthening that factor.

## Post-training evaluation and delivery

For robust08 training seed2, repeat the complete clean/single-factor/combined matrix. For each ablation checkpoint, use clean, the selected isolated factor, and original combined challenge, each with3 evaluation seeds and identical commands. Report per-command counts, per-training-seed summaries, and variation rather than hiding poor seeds in a pooled average. Strict combined-challenge target is <1% for0.3 and0.5 m/s for every evaluation and training seed;0.8 m/s is reported without changing acceptance thresholds. No default promotion unless all required training seeds pass. Deliver colored front/side20-second videos for the empirically worst instruction, with exact precolored asset integrity checks and joint legend. Velocity commands have no HumanML3D NPY source.

Notion record: https://www.notion.so/403325d8231d8172995bf19510b8d760 . Final results and GitHub issue reply follow completed evaluation, then remote commit and push.
