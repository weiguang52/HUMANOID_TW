#!/bin/bash
set -eo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
S=/root/gpufree-data/datasets/practice9/tw123_v1
U=/root/gpufree-data/projects/tw_retargeting/IKRetargeting_V3
test "$(git -C "$U" rev-parse HEAD)" = 655a103e7a0b66dd6afcc31e123bd24e1f478d84
python scripts/practice9/build_tw75_solver_model.py --output "$S/native_training_geometry"
cmake -S "$S/native_training_geometry" -B "$S/native_training_geometry/build/native" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$U/build/deps/cmeel.prefix;/root/gpufree-data/conda_envs/env_isaaclab/lib/python3.11/site-packages/cmeel.prefix"
cmake --build "$S/native_training_geometry/build/native" -j2
ctest --test-dir "$S/native_training_geometry/build/native" --output-on-failure
