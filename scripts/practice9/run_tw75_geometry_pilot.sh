#!/usr/bin/env bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
ROOT=/root/gpufree-data/datasets/practice9/tw75_dataset_v1
VARIANT=${1:-training_geometry_aligned}
MODEL=${2:-$ROOT/native_training_geometry}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo failed > "$ROOT/${VARIANT}.status"' ERR
echo building > "$ROOT/${VARIANT}.status"
PREFIX=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/cmeel.prefix")')
DEPS=/root/gpufree-data/projects/tw_retargeting/IKRetargeting_V3/build/deps/cmeel.prefix
cmake -S "$MODEL" -B "$MODEL/build/native" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$DEPS;$PREFIX"
cmake --build "$MODEL/build/native" -j 2
export LD_LIBRARY_PATH="$DEPS/lib:$PREFIX/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir "$MODEL/build/native" --output-on-failure > "$ROOT/${VARIANT}_native_tests.log" 2>&1 || { echo tests_failed > "$ROOT/${VARIANT}.status"; exit 1; }
echo evaluating > "$ROOT/${VARIANT}.status"
python scripts/practice9/retarget_humanml3d.py --input "$ROOT/pilot.inputs.txt" --output-dir "$ROOT/pilot_${VARIANT}/retargeted" --retarget-root "$MODEL" --retarget-profile "$ROOT/baseline.json" --allow-quality-failures --continue-on-error
echo completed > "$ROOT/${VARIANT}.status"
