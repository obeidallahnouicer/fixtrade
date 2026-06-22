#!/usr/bin/env bash
set -eu

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

SERVICES=(postgres redis auth-service ml-service api genai-service scraper etl-worker)

echo "Starting services: ${SERVICES[*]}" | tee "$LOGDIR/startup.log"
docker compose up -d --build "${SERVICES[@]}" 2>&1 | tee -a "$LOGDIR/startup.log"

wait_for() {
  url=$1
  name=$2
  timeout=${3:-60}
  echo "Waiting for $name at $url" | tee -a "$LOGDIR/startup.log"
  i=0
  until curl -sSf "$url" >/dev/null 2>&1; do
    sleep 2
    i=$((i+2))
    if [ $i -ge $timeout ]; then
      echo "Timeout waiting for $name" | tee -a "$LOGDIR/startup.log"
      return 1
    fi
  done
  echo "$name is available" | tee -a "$LOGDIR/startup.log"
}

wait_for "http://127.0.0.1:5432" "postgres" 120 || true
wait_for "http://127.0.0.1:6379" "redis" 60 || true
wait_for "http://127.0.0.1:8000/api/v1/health" "api" 120 || true
wait_for "http://127.0.0.1:8001/api/v1/health" "ml-service" 120 || true
wait_for "http://127.0.0.1:8002/api/v1/health" "auth-service" 120 || true
wait_for "http://127.0.0.1:8003/api/v1/health" "genai-service" 120 || true

echo "Running ETL loader inside etl-worker (if available)" | tee -a "$LOGDIR/startup.log"
docker compose exec -T etl-worker python scripts/load_fallback_articles.py 2>&1 | tee "$LOGDIR/etl_run.log" || echo "ETL run failed" | tee -a "$LOGDIR/startup.log"

echo "Copying scraped_fallback.jsonl to data/ for persistence if present" | tee -a "$LOGDIR/startup.log"
if [ -f scraped_fallback.jsonl ]; then
  mkdir -p data
  cp scraped_fallback.jsonl data/
  echo "copied" | tee -a "$LOGDIR/startup.log"
fi

echo "Running smoke tests" | tee -a "$LOGDIR/startup.log"
python3 scripts/smoke_test.py "$LOGDIR/smoke_test.json" 2>&1 | tee -a "$LOGDIR/startup.log"

echo "Done. Logs in $LOGDIR" | tee -a "$LOGDIR/startup.log"
