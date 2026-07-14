#!/bin/bash
set -e

LOG_DIR="${OFFER_LOG_DIR:-/app/data/logs}"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/uvicorn_$(date +%Y%m%d).log"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') Starting Offer Ingestion API server..." | tee -a "$LOG_FILE"
echo "  Config : ${OFFER_CONFIG_PATH:-/app/offerIngestion/offer_config.json}" | tee -a "$LOG_FILE"
echo "  ES URL : ${DEFAULT_ES_URL:-not set}" | tee -a "$LOG_FILE"
echo "  Log    : $LOG_FILE" | tee -a "$LOG_FILE"

exec uvicorn api.scheduler_service:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    2>&1 | tee -a "$LOG_FILE"
