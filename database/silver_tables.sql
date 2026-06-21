-- SILVER LAYER: cleaned + typed tables (HAIP, Synthea source)
-- Five connected tables. Keys (id/patient/encounter) kept as TEXT UUIDs
-- so joins work in gold. Dates -> TIMESTAMP, costs -> NUMERIC.

CREATE SCHEMA IF NOT EXISTS silver;

-- 1. Patients — master table, keyed by id
DROP TABLE IF EXISTS silver.patients;
CREATE TABLE silver.patients (
    id                  TEXT,
    birthdate           DATE,
    deathdate           DATE,
    marital             TEXT,
    race                TEXT,
    ethnicity           TEXT,
    gender              TEXT,
    city                TEXT,
    state               TEXT,
    county              TEXT,
    zip                 TEXT,
    healthcare_expenses NUMERIC,
    healthcare_coverage NUMERIC,
    income              NUMERIC,
    _loaded_at          TIMESTAMP DEFAULT now()
);

-- 2. Encounters — admissions + cost. patient -> patients.id
DROP TABLE IF EXISTS silver.encounters;
CREATE TABLE silver.encounters (
    id                  TEXT,
    start               TIMESTAMP,
    stop                TIMESTAMP,
    patient             TEXT,
    organization        TEXT,
    payer               TEXT,
    encounterclass      TEXT,
    code                TEXT,
    description         TEXT,
    base_encounter_cost NUMERIC,
    total_claim_cost    NUMERIC,
    payer_coverage      NUMERIC,
    reasoncode          TEXT,
    reasondescription   TEXT,
    _loaded_at          TIMESTAMP DEFAULT now()
);

-- 3. Conditions — diagnoses. patient + encounter
DROP TABLE IF EXISTS silver.conditions;
CREATE TABLE silver.conditions (
    start       DATE,
    stop        DATE,
    patient     TEXT,
    encounter   TEXT,
    system      TEXT,
    code        TEXT,
    description TEXT,
    _loaded_at  TIMESTAMP DEFAULT now()
);

-- 4. Observations — labs + vitals. patient + encounter
DROP TABLE IF EXISTS silver.observations;
CREATE TABLE silver.observations (
    date        TIMESTAMP,
    patient     TEXT,
    encounter   TEXT,
    category    TEXT,
    code        TEXT,
    description TEXT,
    value       TEXT,
    units       TEXT,
    type        TEXT,
    _loaded_at  TIMESTAMP DEFAULT now()
);

-- 5. Procedures — operations. patient + encounter
DROP TABLE IF EXISTS silver.procedures;
CREATE TABLE silver.procedures (
    start             TIMESTAMP,
    stop              TIMESTAMP,
    patient           TEXT,
    encounter         TEXT,
    system            TEXT,
    code              TEXT,
    description       TEXT,
    base_cost         NUMERIC,
    reasoncode        TEXT,
    reasondescription TEXT,
    _loaded_at        TIMESTAMP DEFAULT now()
);

-- Grants for haip_user
GRANT USAGE ON SCHEMA silver TO haip_user;
GRANT ALL ON ALL TABLES IN SCHEMA silver TO haip_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA silver GRANT ALL ON TABLES TO haip_user;