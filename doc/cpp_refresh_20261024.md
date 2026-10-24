# cpp refresh, 2026-10-24

Upstream: weiguang52/tw_retargeting, cpp,
00d4317f6dbd2a64140346c68c916759d0e4fd75.
Remote checkout: /root/gpufree-data/projects/tw_retargeting_cpp_latest.
Old checkout and original trained artifacts are retained.

The generator now accepts --upstream-root and handles the new model-bound clamp.
A version-guarded parentJoint/parent expression supports Pinocchio 3 and the
installed 2.7 without replacing the Isaac environment. The derived solver retains
training URDF geometry/limits and the upstream selected_head_axis profile.
Upstream Assembly hip/shoulder ranges are not substituted for robot limits.
Experimental shoulder-sector and arm-branch profiles are not enabled because they
assume Assembly shoulder geometry. This is an intentional training-model adaptation.

Validation: upstream geometry with only the Pinocchio API compatibility patch
passes all 10 CTests. Derived training geometry passes seven compatible CTests;
upstream core limit assertions, arm-branch geometry and shoulder-sector geometry
checks do not apply to its different model and fail. Derived FK agreement was
verified on 100 configurations with zero matrix error. Three refresh regression
tests pass, including a real clip and QP residual <=1e-6. Existing 102 Python tests
pass when run per file; running all in one process triggers the pre-existing
CPU-device-guard test because another test already imported torch.

Pipeline: scripts/practice9/build_cpp_refresh.sh, then
scripts/practice9/run_cpp_refresh.sh in tmux cpp_refresh_20261024.
State: /root/gpufree-data/datasets/practice9/cpp_refresh_20261024.
Regenerate the exact 1348-candidate TW215 train selection using the pinned solver,
apply the existing TW123 near-limit advisory policy (no clipping), simulator FK,
TW195 contact projection/smoothing and unchanged strict projection gates.
Rebuild HumanML3D contact labels for new frame counts; rebalance categories.
Require at least 300 walking and 1000 standing-upper clips before training.
No previous FK arrays are reused for changed joint trajectories.

Continue the TW215 neck-only model_23997 checkpoint on the unchanged action
coordinate contract and task architecture. First 64 environments/2 iterations,
then 1024 environments/2000 additional iterations, GPU1, seed42. Outputs have
new run names and state directories. A failed data/smoke gate prevents training.
Training completion and held-out quality validation remain separate milestones.
