#!/bin/bash
set -e
cd /home/ubuntu/CAPSTONE-HEALTHCARE-PROJECT

# Free disk before pulling new images (prevents "no space left")
docker system prune -af || true

docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d --force-recreate
