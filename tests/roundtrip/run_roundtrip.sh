#!/usr/bin/env bash
# Full baler round trip (train -> compress -> decompress -> plot) on a bundled
# project with its config unchanged, followed by tests/roundtrip/check_roundtrip.py.
#
# Run from the repository root (baler resolves workspaces/... relative to CWD):
#     tests/roundtrip/run_roundtrip.sh [WORKSPACE] [PROJECT]
# Uses whatever `python` is on PATH; under uv: `uv run tests/roundtrip/run_roundtrip.sh`.
#
# Writes to $ROUNDTRIP_OUT (default: roundtrip_results/):
#   timings.json, roundtrip_metrics.json, <mode>.log
# If ROUNDTRIP_SUMMARY_MD is set, a markdown summary is appended to that file
# (CI sets it to $GITHUB_STEP_SUMMARY).
set -euo pipefail

WORKSPACE="${1:-CMS_workspace}"
PROJECT="${2:-CMS_project_v1}"
OUT_DIR="${ROUNDTRIP_OUT:-roundtrip_results}"
PYTHON="${PYTHON:-python}"

if [[ ! -d "workspaces/${WORKSPACE}/${PROJECT}" ]]; then
    echo "error: workspaces/${WORKSPACE}/${PROJECT} not found; run from the repo root" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"
timings="${OUT_DIR}/timings.json"
echo "{" > "${timings}"

total_start=$(date +%s)
sep=""
status=0
for mode in train compress decompress plot; do
    echo "=== baler --mode ${mode} ==="
    start=$(date +%s)
    # Capture the status instead of letting set -e exit here, so timings.json is
    # still closed (valid JSON) when a mode fails.
    "${PYTHON}" -m baler --project "${WORKSPACE}" "${PROJECT}" --mode "${mode}" 2>&1 \
        | tee "${OUT_DIR}/${mode}.log" || status=$?
    end=$(date +%s)
    echo "=== ${mode}: $((end - start)) s ==="
    printf '%s  "%s": %d' "${sep}" "${mode}" "$((end - start))" >> "${timings}"
    sep=$',\n'
    if [[ ${status} -ne 0 ]]; then
        printf ',\n  "failed_mode": "%s"' "${mode}" >> "${timings}"
        break
    fi
done
total_end=$(date +%s)
printf ',\n  "total": %d\n}\n' "$((total_end - total_start))" >> "${timings}"

echo "=== per-step wall time (s) ==="
cat "${timings}"

if [[ ${status} -ne 0 ]]; then
    echo "error: baler --mode ${mode} failed (exit ${status}); see ${OUT_DIR}/${mode}.log" >&2
    exit "${status}"
fi

summary_args=()
if [[ -n "${ROUNDTRIP_SUMMARY_MD:-}" ]]; then
    summary_args=(--summary-md "${ROUNDTRIP_SUMMARY_MD}")
fi

"${PYTHON}" tests/roundtrip/check_roundtrip.py \
    --workspace "${WORKSPACE}" --project "${PROJECT}" \
    --metrics-out "${OUT_DIR}/roundtrip_metrics.json" \
    --timings "${timings}" ${summary_args[@]+"${summary_args[@]}"}
