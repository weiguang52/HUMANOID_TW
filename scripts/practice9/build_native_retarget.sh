#!/usr/bin/env bash
set -euo pipefail
DATA="${GPUFREE_DATA_ROOT:-/root/gpufree-data}"
ROOT="${PRACTICE9_RETARGET_ROOT:-$DATA/projects/tw_retargeting/IKRetargeting_V3}"
PREFIX="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/cmeel.prefix")')"
test -f "$ROOT/CMakeLists.txt" || { echo "Clone weiguang52/tw_retargeting into $DATA/projects/tw_retargeting first." >&2; exit 2; }
test -f "$PREFIX/lib/cmake/pinocchio/pinocchioConfig.cmake" || { echo "C++ Pinocchio missing from $PREFIX" >&2; exit 2; }
if [[ ! -d "$ROOT/build/deps/cmeel.prefix/include/Eigen" ]]; then
  python -m pip install --no-deps --target "$ROOT/build/deps" cmeel-eigen==3.4.1 cmeel-urdfdom-headers==3.0.0
fi
cmake -S "$ROOT" -B "$ROOT/build/native" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ROOT/build/deps/cmeel.prefix;$PREFIX"
cmake --build "$ROOT/build/native" -j "${P9_BUILD_JOBS:-2}"
LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}" ctest --test-dir "$ROOT/build/native" --output-on-failure
