#!/bin/bash
set -e
cd /home/ubuntu/CAPSTONE-HEALTHCARE-PROJECT

AWS_REGION=eu-north-1
ACCOUNT_ID=700640309357

# Authenticate Docker to ECR (EC2 role provides credentials)
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d --force-recreate
