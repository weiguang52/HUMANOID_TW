# Server migration support files

Copied remote experiment orchestration and audit source, not datasets or model artifacts. These helpers were originally run within state directories under /root/gpufree-data/twrl_runs; restore each helper to its matching state directory before use. They retain original absolute paths and must be reviewed for new server layout. No training was started during migration.

Read docs/practice9/handoff_20261029.md first. Git alone does NOT include HumanML3D motion NPZ/manifests, robot URDF/USD/meshes, original27995 and candidate checkpoints, immutable teacher policy, state evidence or videos. Transfer those separately directly server-to-server before running. Preserve hashes and directory layout/symlinks. Install IsaacLab/IsaacSim and matching conda environment; do not assume Git supplies these.

Required source roots: /root/gpufree-data/datasets/practice9 (especially training_speed_blend_20261026, head_locked_20261024, frozen_teachers, model assets and referenced motion sources), /root/gpufree-data/twrl_runs (experiment states and logs), and the original checkpoint path under repository logs shown in handoff. Dependencies may be absolute paths outside these roots; resolve manifest and URDF references before migration validation. The /root/twrl_runs alias must target the new data directory.

Current prepared orientation experiment has only smoke and unit checks completed; reload/deployment/formal/evaluation remain. Formal scripts intentionally refuse existing output status. Do not automatically launch or overwrite runs.
