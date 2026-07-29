# HAIP — Healthcare Analytics & Intelligence Platform

An end-to-end platform that lets a hospital securely upload its own data and instantly receive analytics, machine-learning predictions, admissions forecasts, and two generative-AI features — a document-grounded chatbot and a tool-calling clinical agent. Built as six containerised microservices behind an nginx reverse proxy and deployed to AWS through a fully automated CI/CD pipeline.

**Live:** http://med-haip.duckdns.org
**API docs:** http://med-haip.duckdns.org/api/docs

---

## Overview

Hospital data is scattered across disconnected systems, and there is no single tool that turns it into both analytics and forward-looking intelligence. HAIP closes that gap. A hospital logs in, uploads its data, and immediately gets validated KPIs and charts, a machine-learning model it trains on its own data, an admissions forecast that feeds a staffing plan, a provider analytics view, a retrieval-augmented chatbot over its own documents, and an AI agent that reasons across its live metrics — all behind one secure interface, with strict per-hospital data isolation.

The project covers the full data-science and engineering lifecycle: data engineering, data quality and validation, machine learning, time-series forecasting, retrieval-augmented generation, agentic AI, a full-stack web application, and an automated cloud deployment.

---

## Key Features

- **Secure multi-tenant access** — JWT authentication with strict per-hospital data isolation.
- **Smart data upload** — automatic column mapping, validation, and per-column data-quality reporting before any analysis.
- **Analytics & KPIs** — mortality rate, readmission, DAMA rate, comorbidity burden, revenue trends, and exploratory charts.
- **Machine learning** — user-trained Random Forest (classification or regression, auto-detected) with a leak-free scikit-learn pipeline and honest metrics plus feature importances.
- **Forecasting** — Holt's additive-trend exponential smoothing, translated into a staffing/capacity plan.
- **Provider analytics** — department- and provider-level performance across 941 providers.
- **RAG chatbot** — per-hospital document Q&A with two-stage retrieval (FAISS bi-encoder + cross-encoder re-ranking) and cited, grounded answers.
- **AI agent** — a LangGraph ReAct agent with four whitelisted tools (guidelines, KPIs, forecasts, doctor stats).
- **Automated CI/CD** — every push tests, builds, and deploys to a live cloud server with no manual steps.

---

## Architecture

HAIP runs as six containerised services behind a single nginx reverse proxy. nginx is the only public entry point (port 80); it routes each request by URL path and keeps the backend services private.

| Service | Tech | Port | Role |
|---|---|---|---|
| frontend | React (Vite build) + nginx | 80 | Shell + reverse proxy (public entry point) |
| fastapi | FastAPI | 8000 | JWT auth and routing |
| merged_app | Dash | 8060 | Analytics dashboard (KPIs, ML, forecast) |
| doctor_app | Dash | 8051 | Provider/doctor analytics |
| streamlit | Streamlit | 8501 | RAG document chat |
| agent_app | FastAPI + LangGraph | 8062 | Tool-calling AI agent |

The five Python services run from a single shared Docker image and differ only in their start command. The React frontend embeds the three dashboards as iframes and implements the agent chat natively, using relative API paths so the same build runs unchanged across environments.

---

## Tech Stack

- **Languages:** Python, JavaScript (React), SQL
- **Backend:** FastAPI, Dash, Streamlit
- **ML / Forecasting:** scikit-learn (Random Forest), statsmodels (exponential smoothing)
- **GenAI:** LangChain, LangGraph, sentence-transformers, FAISS; Gemini / Groq LLMs
- **Frontend:** React (Vite), served by nginx
- **Storage:** SQLite (auth), CSV (hospital data), FAISS (vector indexes)
- **Infrastructure:** Docker, Docker Compose, nginx reverse proxy
- **CI/CD:** GitHub Actions, GitHub Container Registry (GHCR)
- **Cloud:** AWS EC2 (Ubuntu), Elastic IP, DuckDNS

---

## CI/CD Pipeline

On every push to \`main\`: CI runs syntax checks and unit tests; if it passes, CD builds the backend and frontend Docker images, pushes them to GHCR, then connects to the EC2 server over SSH, pulls the new images, and restarts the containers — updating the live site with no manual steps. Images are built on GitHub's runners so the live site is never slowed by a build. A full AWS-native pipeline (ECR, CodeBuild, CodeDeploy) is also designed and committed.

---

## Local Setup

\`\`\`bash
git clone https://github.com/Sooryasooru/CAPSTONE-HEALTHCARE-PROJECT.git
cd CAPSTONE-HEALTHCARE-PROJECT
cp .env.example .env        # then fill in the required values
docker compose up -d --build
# app available at http://localhost
\`\`\`

---

## Data & Privacy

HAIP uses synthetic, de-identified data — real hospital datasets are difficult to obtain for privacy reasons. The project began with Synthea for early demonstration, then moved to purpose-built structured datasets modelling realistic hospital, doctor, and department relationships. No real patient data is used.

---

## Testing

Unit tests cover retrieval metrics, KPI calculations, forecasting, agent tools, and API endpoints, and run automatically in CI on every push. Deployment proceeds only if the tests pass.

---

## Roadmap

- Managed database (PostgreSQL) and managed vector store for scale
- Activate the AWS-native CI/CD pipeline
- HTTPS with a managed certificate, monitoring, and centralised logging
- Real-time data streaming and EHR integration
- Richer agent tools and additional predictive models

---

## Author

**Soorya** — Data Science Capstone (Brototype)
GitHub: [@Sooryasooru](https://github.com/Sooryasooru)
