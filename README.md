# 🏥 Healthcare Analytics & Intelligence Platform (HAIP)

> An end-to-end healthcare data platform that consolidates fragmented hospital data into a single, governed, analytics-ready foundation for **analytics, machine learning, and Retrieval-Augmented Generation (RAG)**.

![Status](https://img.shields.io/badge/Phase-Week%201%3A%20Data%20Foundation-blue)
![Database](https://img.shields.io/badge/Database-PostgreSQL-336791)
![Python](https://img.shields.io/badge/Python-3.x-3776AB)
![Architecture](https://img.shields.io/badge/Architecture-Medallion-orange)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Data Sources](#-data-sources)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Running the Pipeline](#-running-the-pipeline)
- [EDA Highlights](#-eda-highlights)
- [Deliverables](#-deliverables)
- [Roadmap](#-roadmap)

---

## 🎯 Overview

Hospital data typically lives in disconnected systems — admissions, billing, ICU monitoring, and laboratory results each use different identifiers, formats, and conventions. This fragmentation prevents unified analysis and blocks predictive modeling.

**HAIP** addresses this by building a single trustworthy data layer with explicit data-quality guarantees, processed through a layered **medallion architecture** (Bronze → Silver → Gold) on PostgreSQL.

### Objectives
- Establish a reliable, reproducible healthcare data foundation.
- Ingest and standardize six heterogeneous data sources.
- Apply a medallion architecture separating raw, cleaned, and analytics-ready data.
- Identify and document missing values, outliers, and data-quality characteristics.
- Validate that cleaned data preserves real clinical relationships.

---

## 🏗 Architecture

```
Raw Sources  →  Bronze  →  Silver  →  Gold  →  Analytics / ML / RAG
                (raw)     (cleaned)  (views)
```

| Layer | Role | Typing | Idempotent |
|-------|------|--------|------------|
| **Bronze** | Raw landing zone — ingest as-is, never fail on bad data | All `TEXT` | Yes (truncate + load) |
| **Silver** | Cleaned, typed, standardized & validated | Real types | Yes (truncate + load) |
| **Gold** | Analytics-ready aggregated views | Views on silver | Rebuilt each run |

> **Design note:** The five tabular sources use inconsistent patient identifiers (`MRD No.`, `Patient_ID`, `subject_id`) and the laboratory source has none. They are therefore treated as **independent analytical domains** and are not force-joined.

Database access uses a dedicated least-privilege role (`haip_user`): **write** access to Bronze/Silver, **read-only** on Gold.

---

## 📊 Data Sources

| # | Dataset | Rows | Cols | Purpose |
|---|---------|------|------|---------|
| 1 | Patients & Admissions | 15,757 | 56 | Demographics, admissions, comorbidities, labs, outcomes |
| 2 | Mortality | 359 | 6 | In-hospital death records |
| 3 | Billing | 984 | 10 | Treatment cost, procedure, satisfaction, readmission |
| 4 | ICU | 5,000 | 77 | Vitals, severity scores, interventions, sepsis labels |
| 5 | Laboratory | 27 | 11 | Lab test results with reference ranges (reference sample) |
| 6 | Medical Documents (RAG) | Corpus | — | Clinical guidelines for retrieval-augmented generation |

**Sourcing rule:** 80% real-world data, 20% synthetic. Medical documents are sourced from `epfl-llm/guidelines` (HuggingFace) and kept **real, never synthetic**.

---

## 📁 Project Structure

```
healthcare-project/
├── config/                     # Configuration files
├── data/
│   ├── raw/                    # Raw source data (CSV + Arrow)
│   ├── processed/              # Intermediate data
│   └── final/                  # Final outputs
├── database/
│   ├── schema.sql              # Master runner (bronze → silver → gold)
│   ├── bronze_tables.sql       # Bronze layer DDL + grants
│   ├── silver_tables.sql       # Silver layer DDL + grants
│   ├── gold_views.sql          # Gold analytics views + grants
│   ├── queries.sql             # Analytical query library
│   └── er_diagram.png          # Entity-relationship diagram
├── etl/
│   ├── config.py               # Paths + DB connection (env-based)
│   ├── utils.py                # Shared logging + engine helpers
│   ├── extract.py              # Read 6 raw sources
│   ├── load.py                 # Load into Bronze (idempotent)
│   ├── transform.py            # Clean + type → Silver
│   └── validate.py             # Data-quality checks
├── flowcharts/                 # Module flowcharts (PNG)
│   ├── data_collection.png
│   ├── eda.png
│   ├── database_design.png
│   └── etl_pipeline.png
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── reports/
│   ├── eda_reports.pdf         # EDA report
│   └── data_dictionary.xlsx    # Column-level documentation
├── tests/                      # Test suite
├── .env                        # Credentials (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

| Tool | Role in HAIP |
|------|--------------|
| **Python 3** | Orchestration language for the entire pipeline |
| **PostgreSQL** | Governed relational store hosting the medallion layers |
| **Pandas** | Data ingestion, cleaning, and transformation |
| **NumPy** | Numerical operations underlying Pandas computations |
| **SQLAlchemy** | Python ↔ PostgreSQL engine and transactions |
| **Matplotlib / Seaborn** | Static EDA visualizations and report figures |
| **HuggingFace `datasets`** | Loading the medical-guideline corpus for RAG |

### Pinned Versions
```
pandas==3.0.3
numpy==2.4.6
SQLAlchemy==2.0.50
psycopg2-binary==2.9.12
matplotlib==3.10.9
seaborn==0.13.2
datasets==5.0.0
python-dotenv
jupyter
ipykernel
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.x
- PostgreSQL 16+

### 1. Clone & create a virtual environment
```bash
git clone <your-repo-url>
cd healthcare-project
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=haip
DB_USER=haip_user
DB_PASSWORD=your_password
```

### 4. Create the database role
```bash
sudo -u postgres psql -d haip -c "CREATE USER haip_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -d haip -c "GRANT ALL PRIVILEGES ON DATABASE haip TO haip_user;"
```

---

## 🚀 Running the Pipeline

```bash
# 1. Build the full medallion schema (bronze → silver → gold)
sudo -u postgres psql -d haip < database/schema.sql

# 2. Load raw data into Bronze
python -m etl.load

# 3. Clean + type into Silver
python -m etl.transform

# 4. Validate data quality
python -m etl.validate
```

Expected validated row counts: **patients 15,757 · mortality 359 · billing 984 · icu 5,000 · labs 27**.

---

## 🔍 EDA Highlights

All EDA findings were validated against established clinical knowledge — confirming both data integrity and correct pipeline transformation.

- **Missing values are informative, not random** — e.g. BNP (57.6% missing) is a selectively-ordered cardiac test; ICU lab panels (~10–11%) are missing in coherent blocks.
- **Outliers are mostly real critical patients** — high creatinine, lactate, BNP, and SOFA scores reflect genuinely sick patients and were retained; only ~0.16% (patients) and ~0.3% (ICU) of values were physiologically implausible and flagged.
- **Correlations match medicine** — creatinine–urea (0.74), creatinine–CKD (0.76), EF–heart failure (−0.38).
- **Mortality stratification** — organ-failure comorbidities triple mortality: AKI 16.2%, CKD 14.4%, heart failure 14.1% vs. chronic conditions ~5%.
- **ICU sepsis escalates with severity** — 2% → 7% → 75% → 100% across SOFA bands.

---

## 📦 Deliverables (Week 1)

- [x] Cleaned healthcare dataset (PostgreSQL Silver layer)
- [x] PostgreSQL database (medallion architecture)
- [x] ETL pipeline (extract / load / transform / validate)
- [x] EDA report (`reports/eda_reports.pdf`)
- [x] Data dictionary (`reports/data_dictionary.xlsx`)
- [x] Module flowcharts (data collection, EDA, database design, ETL)

### Acceptance Criteria
- [x] All datasets collected and documented
- [x] Missing values and outliers identified
- [x] Database schema finalized
- [x] ETL pipeline successfully loads cleaned data into PostgreSQL

---

## 🗺 Roadmap

| Phase | Focus |
|-------|-------|
| ✅ **Week 1** | Data foundation — collection, ETL, medallion DB, EDA |
| 🔜 **Feature Engineering** | Model-ready features (encoding, scaling, missingness handling) |
| 🔜 **Machine Learning** | Mortality & sepsis prediction models on the Gold layer |
| 🔜 **RAG System** | Embeddings + pgvector over the medical-guideline corpus |
| 🔜 **Dashboard** | Interactive Plotly Dash dashboard reading Gold views |

---

## 📝 License & Acknowledgements

Built as a capstone data-engineering project. Datasets sourced from public repositories (Kaggle, HuggingFace) under their respective licenses.


##fro running the dasbaord 
cd ~/Documents/capstone/HAIP/healthcare-project/src
python -m dashboard.app