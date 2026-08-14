#!/usr/bin/env bash
set -euo pipefail
umask 077

# Download licensed AMASS SMPL+H archives used to reconstruct HumanML3D.
# Credentials are deliberately not accepted here. Pass an authenticated
# Netscape cookie jar created on download.is.tue.mpg.de.

DATA_ROOT="${DATA_ROOT:-/root/gpufree-data}"
MANIFEST="${1:-$DATA_ROOT/tmp/humanml3d/amass/sizes.tsv}"
COOKIE_JAR="${2:-$DATA_ROOT/tmp/humanml3d/amass/download_cookies.txt}"
OUTPUT_DIR="${3:-$DATA_ROOT/datasets/humanml3d/amass_archives}"
JOBS="${HUMANML3D_DOWNLOAD_JOBS:-3}"
LOG_DIR="${HUMANML3D_DOWNLOAD_LOG_DIR:-$DATA_ROOT/tmp/humanml3d/download_logs}"
AUTH_CONFIG="${HUMANML3D_AUTH_CONFIG:-}"
AUTH_RETRY_DELAY="${HUMANML3D_AUTH_RETRY_DELAY:-300}"
AUTH_RETRY_ATTEMPTS="${HUMANML3D_AUTH_RETRY_ATTEMPTS:-72}"

if [[ ! -s "$MANIFEST" ]]; then
  echo "Missing archive manifest: $MANIFEST" >&2
  exit 2
fi
if [[ ! -s "$COOKIE_JAR" ]]; then
  echo "Missing authenticated AMASS cookie jar: $COOKIE_JAR" >&2
  exit 2
fi
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "HUMANML3D_DOWNLOAD_JOBS must be a positive integer." >&2
  exit 2
fi
if ! [[ "$AUTH_RETRY_DELAY" =~ ^[1-9][0-9]*$ && "$AUTH_RETRY_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Authentication retry delay and attempts must be positive integers." >&2
  exit 2
fi
if [[ -n "$AUTH_CONFIG" && ! -s "$AUTH_CONFIG" ]]; then
  echo "Missing curl authentication config: $AUTH_CONFIG" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
chmod 700 "$OUTPUT_DIR" "$LOG_DIR"

download_one() {
  local name="$1"
  local expected_size="$2"
  local url="$3"
  local final="$OUTPUT_DIR/${name}.tar.bz2"
  local partial="${final}.part"
  local log="$LOG_DIR/${name}.log"

  if [[ -f "$final" ]] && [[ "$(stat -c '%s' "$final")" == "$expected_size" ]]; then
    bzip2 -t "$final"
    printf 'READY\t%s\t%s\n' "$name" "$expected_size"
    return
  fi

  {
    printf 'START\t%s\t%s\n' "$name" "$(date --iso-8601=seconds)"
    local cookie="$COOKIE_JAR"
    if [[ -n "$AUTH_CONFIG" ]]; then
      local worker_cookie="$LOG_DIR/.${name}.cookies.txt"
      local status=""
      local attempt
      rm -f "$worker_cookie"
      for ((attempt = 1; attempt <= AUTH_RETRY_ATTEMPTS; attempt++)); do
        status="$(
          curl --config "$AUTH_CONFIG" --silent --show-error \
            --cookie-jar "$worker_cookie" --output /dev/null \
            --write-out '%{http_code}' "$url" || true
        )"
        if [[ "$status" == "302" ]]; then
          break
        fi
        printf 'AUTH_RETRY\t%s\t%s/%s\thttp=%s\n' \
          "$name" "$attempt" "$AUTH_RETRY_ATTEMPTS" "$status"
        sleep "$AUTH_RETRY_DELAY"
      done
      if [[ "$status" != "302" ]]; then
        echo "Authentication did not recover for $name." >&2
        return 6
      fi
      chmod 600 "$worker_cookie"
      cookie="$worker_cookie"
    fi

    if ! curl --fail --location --continue-at - \
      --cookie "$cookie" \
      --connect-timeout 30 \
      --retry 30 --retry-delay 5 --retry-all-errors \
      --speed-time 300 --speed-limit 1024 \
      --output "$partial" "$url"; then
      if [[ -f "$partial" && "$(stat -c '%s' "$partial")" -le 16384 ]] \
        && head -c 15 "$partial" | grep -q '<!DOCTYPE html>'; then
        rm -f "$partial"
      fi
      return 7
    fi

    if [[ "$(head -c 3 "$partial")" != "BZh" ]]; then
      echo "Invalid archive signature for $name; removing response." >&2
      rm -f "$partial"
      return 8
    fi

    local actual_size
    actual_size="$(stat -c '%s' "$partial")"
    if [[ "$actual_size" != "$expected_size" ]]; then
      echo "Size mismatch for $name: expected $expected_size, got $actual_size" >&2
      return 4
    fi
    bzip2 -t "$partial"
    mv "$partial" "$final"
    printf 'DONE\t%s\t%s\t%s\n' "$name" "$actual_size" "$(date --iso-8601=seconds)"
  } >> "$log" 2>&1
}

export -f download_one
export OUTPUT_DIR LOG_DIR COOKIE_JAR AUTH_CONFIG AUTH_RETRY_DELAY AUTH_RETRY_ATTEMPTS

running=0
failed=0
while IFS=$'\t' read -r name expected_size url; do
  [[ -n "$name" ]] || continue
  download_one "$name" "$expected_size" "$url" &
  ((running += 1))
  if (( running >= JOBS )); then
    if ! wait -n; then failed=1; fi
    ((running -= 1))
  fi
done < "$MANIFEST"

while (( running > 0 )); do
  if ! wait -n; then failed=1; fi
  ((running -= 1))
done

if (( failed != 0 )); then
  echo "One or more downloads failed. Re-run this script to resume." >&2
  exit 5
fi

echo "All AMASS archives are downloaded and verified in $OUTPUT_DIR"
