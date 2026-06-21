-- BRONZE LAYER: raw landing tables (HAIP, Synthea source)
-- All TEXT; cleaning + typing happens in silver.
-- Five connected tables. patients.id is the master key;
-- every other table carries patient (+ encounter) to preserve joins.

CREATE SCHEMA IF NOT EXISTS bronze;

-- 1. Patients (28 cols) — master patient table
DROP TABLE IF EXISTS bronze.patients;
CREATE TABLE bronze.patients (
    id TEXT, birthdate TEXT, deathdate TEXT, ssn TEXT, drivers TEXT,
    passport TEXT, prefix TEXT, first TEXT, middle TEXT, last TEXT,
    suffix TEXT, maiden TEXT, marital TEXT, race TEXT, ethnicity TEXT,
    gender TEXT, birthplace TEXT, address TEXT, city TEXT, state TEXT,
    county TEXT, fips TEXT, zip TEXT, lat TEXT, lon TEXT,
    healthcare_expenses TEXT, healthcare_coverage TEXT, income TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- 2. Encounters (15 cols) — admissions + cost. patient -> patients.id
DROP TABLE IF EXISTS bronze.encounters;
CREATE TABLE bronze.encounters (
    id TEXT, start TEXT, stop TEXT, patient TEXT, organization TEXT,
    provider TEXT, payer TEXT, encounterclass TEXT, code TEXT,
    description TEXT, base_encounter_cost TEXT, total_claim_cost TEXT,
    payer_coverage TEXT, reasoncode TEXT, reasondescription TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- 3. Conditions (7 cols) — diagnoses. patient + encounter
DROP TABLE IF EXISTS bronze.conditions;
CREATE TABLE bronze.conditions (
    start TEXT, stop TEXT, patient TEXT, encounter TEXT,
    system TEXT, code TEXT, description TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- 4. Observations (9 cols) — labs + vitals. patient + encounter
DROP TABLE IF EXISTS bronze.observations;
CREATE TABLE bronze.observations (
    date TEXT, patient TEXT, encounter TEXT, category TEXT, code TEXT,
    description TEXT, value TEXT, units TEXT, type TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- 5. Procedures (10 cols) — operations. patient + encounter
DROP TABLE IF EXISTS bronze.procedures;
CREATE TABLE bronze.procedures (
    start TEXT, stop TEXT, patient TEXT, encounter TEXT, system TEXT,
    code TEXT, description TEXT, base_cost TEXT, reasoncode TEXT,
    reasondescription TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- Grants for haip_user
GRANT USAGE ON SCHEMA bronze TO haip_user;
GRANT ALL ON ALL TABLES IN SCHEMA bronze TO haip_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA bronze GRANT ALL ON TABLES TO haip_user;