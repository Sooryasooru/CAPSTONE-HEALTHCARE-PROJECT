#!/bin/bash
# ApplicationStop hook: bring down running containers if any.
cd /home/ubuntu/CAPSTONE-HEALTHCARE-PROJECT || exit 0
docker compose -f docker-compose.prod.yml down || true
