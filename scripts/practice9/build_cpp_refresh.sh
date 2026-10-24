#!/bin/bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
UPSTREAM=/root/gpufree-data/projects/tw_retargeting_cpp_latest/IKRetargeting_V3
STATE=/root/gpufree-data/datasets/practice9/cpp_refresh_20261024
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == 00d4317f6dbd2a64140346c68c916759d0e4fd75 ]]
python scripts/practice9/build_tw75_solver_model.py --upstream-root "$UPSTREAM" --output "$STATE/native_training_geometry"
cmake -S "$STATE/native_training_geometry" -B "$STATE/native_training_geometry/build/native" -DCMAKE_BUILD_TYPE=Release '-DCMAKE_PREFIX_PATH=/root/gpufree-data/projects/tw_retargeting/IKRetargeting_V3/build/deps/cmeel.prefix;/root/gpufree-data/conda_envs/env_isaaclab/lib/python3.11/site-packages/cmeel.prefix'
cmake --build "$STATE/native_training_geometry/build/native" -j2
python -m unittest discover -s tests/practice9 -p test_cpp_refresh.py
