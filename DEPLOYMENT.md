# HAIP — Deployment Guide

HAIP is a 6-service stack orchestrated with Docker Compose:

| Service    | Port | Tech                 |
|------------|------|----------------------|
| frontend   | 5173 | React / Vite         |
| fastapi    | 8000 | FastAPI (router/auth)|
| merged_app | 8060 | Dash (Card 1)        |
| doctor_app | 8051 | Dash (Card 2)        |
| streamlit  | 8501 | Streamlit (Card 3 RAG)|
| agent_app  | 8062 | LangGraph agent      |

## Resource requirement

The stack loads PyTorch (CPU), FAISS (6,832 vectors), and sentence-transformer
models across the Python services, requiring **~8 GB RAM**. AWS free tier
(t2.micro, 1 GB) is insufficient and will OOM on boot; a full cloud deployment
needs a **t3.large** (8 GB) or equivalent. The stack is fully containerized, so
it runs identically on any adequately-provisioned host.

## Local deployment (demo)

```bash
cp .env.example .env          # add GEMINI_API_KEY, HAIP_JWT_SECRET
./deploy.sh                   # build + start, waits for /health
```

Open http://localhost:5173.

## AWS EC2 deployment (production-style)

### 1. Launch instance
- AMI: Ubuntu 22.04 LTS
- Type: **t3.large** (8 GB RAM)
- Storage: 30 GB gp3 (torch images are large)
- Security group inbound: 22 (SSH), 5173, 8000, 8060, 8051, 8501, 8062

### 2. Install Docker
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER && newgrp docker
```

### 3. Clone + configure
```bash
git clone https://github.com/Sooryasooru/CAPSTONE-HEALTHCARE-PROJECT.git
cd CAPSTONE-HEALTHCARE-PROJECT
nano .env        # GEMINI_API_KEY=...  HAIP_JWT_SECRET=...
```

### 4. Deploy
```bash
./deploy.sh
```

Access at `http://<EC2_PUBLIC_IP>:5173`.

### 5. Operational notes
- `docker compose ps` — service status
- `docker compose logs -f <service>` — tail logs
- `./deploy.sh restart` — restart without rebuild
- **Never** `docker compose down` — wipes ephemeral SQLite/FAISS state
- merged_app has no source mount → rebuild after edits; doctor_app + streamlit
  have `./src` mounts → restart suffices

### 6. Cost control
t3.large bills ~hourly. For a demo, run during the session and
`terminate` afterward to avoid ongoing charges.

## CI/CD

- **CI** (`.github/workflows/ci.yml`): syntax check + pytest on every push.
- **CD** (`.github/workflows/cd.yml`): builds backend + frontend Docker images
  after CI passes, validating the artifacts are deployable.
