#!/bin/bash
set -eo pipefail
GPU=${1:?}
cd /root/gpufree-data/projects/HUMANOID_TW
case "$GPU" in
 0) VARIANTS=(neck baseline);;
 1) VARIANTS=(root precision);;
 *) exit 2;;
esac
STATE=/root/gpufree-data/datasets/practice9/tw202_v1
exec 7>"$STATE/worker$GPU.lock"; flock -n 7
trap 'echo failed > "$STATE/worker$GPU.status"' ERR
for V in "${VARIANTS[@]}"; do
 echo "training_$V" > "$STATE/worker$GPU.status"
 bash scripts/practice9/train_tw202.sh "$GPU" "$V" 2000 formal
done
for V in "${VARIANTS[@]}"; do
 echo "validation_$V" > "$STATE/worker$GPU.status"
 bash scripts/practice9/validate_tw202.sh "$GPU" "$V"
done
echo completed > "$STATE/worker$GPU.status"
