#!/usr/bin/env bash
# HAIP one-command deploy — brings the full 6-service stack up via Docker Compose.
# Usage: ./deploy.sh          (build + start)
#        ./deploy.sh restart  (restart without rebuild)
set -euo pipefail

if [ ! -f .env ]; then
  echo "ERROR: .env missing. Create it with GEMINI_API_KEY and HAIP_JWT_SECRET."
  exit 1
fi

MODE="${1:-up}"

if [ "$MODE" = "restart" ]; then
  echo ">> Restarting services (no rebuild)..."
  docker compose restart
else
  echo ">> Building and starting HAIP stack..."
  docker compose up -d --build
fi

echo ">> Waiting for API health check..."
for i in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    echo ">> API healthy."
    break
  fi
  sleep 3
done

echo ""
echo "HAIP is up:"
echo "  Frontend   : http://localhost:5173"
echo "  FastAPI    : http://localhost:8000/health"
echo "  Card1 Dash : http://localhost:8060"
echo "  Card2 Dash : http://localhost:8051"
echo "  Streamlit  : http://localhost:8501"
echo "  Agent      : http://localhost:8062"
