#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

NSYS_DIR="${RESULTS_ROOT}/nsys"
TRACE_NAME="${TRACE_NAME:-qkv-norm-rope-vnorm-kvcache-fusion}"

REPORT_PREFIX="${NSYS_DIR}/${TRACE_NAME}"
REPORT_FILE="${REPORT_PREFIX}.nsys-rep"
SQLITE_FILE="${REPORT_PREFIX}.sqlite"
STATS_PREFIX="${REPORT_PREFIX}_stats"
NVTX_EVENTS_TXT="${REPORT_PREFIX}_nvtx_events.txt"
ATTENTION_PREP_NVTX_TXT="${REPORT_PREFIX}_attention_prep_nvtx.txt"

if [[ ! -f "$REPORT_FILE" ]]; then
  echo "Missing nsys report: ${REPORT_FILE}" >&2
  exit 1
fi

mkdir -p "$NSYS_DIR"

echo "Processing ${REPORT_FILE}"
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum,osrt_sum \
  --force-overwrite true \
  --format csv,column \
  --output "${STATS_PREFIX}" \
  "$REPORT_FILE"

if [[ -f "${SQLITE_FILE}" ]]; then
  echo "Exporting raw NVTX events from ${SQLITE_FILE}"
  sqlite3 "${SQLITE_FILE}" "
    SELECT
      start,
      end,
      COALESCE(text, (SELECT value FROM StringIds WHERE id = textId)) AS name
    FROM NVTX_EVENTS
    WHERE COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))
      IS NOT NULL
    ORDER BY start;
  " > "${NVTX_EVENTS_TXT}"

  sqlite3 "${SQLITE_FILE}" "
    SELECT
      start,
      end,
      COALESCE(text, (SELECT value FROM StringIds WHERE id = textId)) AS name
    FROM NVTX_EVENTS
    WHERE COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))
      LIKE 'gemma4.attention.prep:%'
    ORDER BY start;
  " > "${ATTENTION_PREP_NVTX_TXT}"
else
  echo "Skipping raw NVTX export because ${SQLITE_FILE} is missing"
fi

echo "Generated:"
echo "  ${STATS_PREFIX}.csv"
echo "  ${STATS_PREFIX}.txt"
if [[ -f "${NVTX_EVENTS_TXT}" ]]; then
  echo "  ${NVTX_EVENTS_TXT}"
fi
if [[ -f "${ATTENTION_PREP_NVTX_TXT}" ]]; then
  echo "  ${ATTENTION_PREP_NVTX_TXT}"
fi
