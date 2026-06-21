-- GOLD LAYER: analytics-ready views (HAIP, Synthea source)
-- Built on silver. Unlike the old independent sources, these views
-- JOIN across tables on the shared patient / encounter keys.

CREATE SCHEMA IF NOT EXISTS gold;

-- 1. Admissions summary by month + encounter class
DROP VIEW IF EXISTS gold.admissions_summary;
CREATE VIEW gold.admissions_summary AS
SELECT
    date_trunc('month', start)                       AS admission_month,
    encounterclass,
    count(*)                                         AS total_admissions,
    round(avg(EXTRACT(EPOCH FROM (stop - start)) / 86400.0)::numeric, 2)
                                                     AS avg_stay_days,
    round(avg(total_claim_cost), 2)                  AS avg_cost
FROM silver.encounters
WHERE start IS NOT NULL
GROUP BY admission_month, encounterclass
ORDER BY admission_month, encounterclass;

-- 2. Patient outcome distribution (alive vs deceased) — uses patients
DROP VIEW IF EXISTS gold.outcome_distribution;
CREATE VIEW gold.outcome_distribution AS
SELECT
    CASE WHEN deathdate IS NULL THEN 'Alive' ELSE 'Deceased' END AS outcome,
    count(*)                                                     AS patients,
    round(100.0 * count(*) / sum(count(*)) OVER (), 1)          AS pct
FROM silver.patients
GROUP BY outcome
ORDER BY patients DESC;

-- 3. Comorbidity prevalence — JOINS conditions to patients
--    Counts distinct patients per common chronic condition.
DROP VIEW IF EXISTS gold.comorbidity_prevalence;
CREATE VIEW gold.comorbidity_prevalence AS
WITH total AS (SELECT count(*) AS n FROM silver.patients)
SELECT
    c.description                                        AS condition,
    count(DISTINCT c.patient)                            AS cases,
    round(100.0 * count(DISTINCT c.patient) / (SELECT n FROM total), 1)
                                                         AS prevalence_pct
FROM silver.conditions c
WHERE c.description IN (
    'Diabetes', 'Hypertension', 'Coronary Heart Disease',
    'Chronic kidney disease stage 1 (disorder)', 'Heart failure (disorder)',
    'Prediabetes', 'Anemia (disorder)'
)
GROUP BY c.description
ORDER BY cases DESC;

-- 4. Cost by condition — JOINS conditions to encounters on encounter key
DROP VIEW IF EXISTS gold.cost_by_condition;
CREATE VIEW gold.cost_by_condition AS
SELECT
    c.description                          AS condition,
    count(DISTINCT c.patient)              AS patients,
    count(e.id)                            AS encounters,
    round(avg(e.total_claim_cost), 2)      AS avg_encounter_cost,
    round(sum(e.total_claim_cost), 2)      AS total_cost
FROM silver.conditions c
JOIN silver.encounters e ON c.encounter = e.id
GROUP BY c.description
HAVING count(DISTINCT c.patient) >= 5
ORDER BY total_cost DESC;
-- 5. Patient 360 base — one row per patient, journey rolled up across tables
--    Pre-aggregates each table to per-patient summaries BEFORE joining,
--    so there is no join explosion. JOINS patients + encounters
--    + conditions + procedures.
DROP VIEW IF EXISTS gold.patient_360;
CREATE VIEW gold.patient_360 AS
WITH enc AS (
    SELECT patient,
           count(*)               AS total_encounters,
           sum(total_claim_cost)  AS lifetime_cost
    FROM silver.encounters
    GROUP BY patient
),
cond AS (
    SELECT patient, count(DISTINCT code) AS distinct_conditions
    FROM silver.conditions
    GROUP BY patient
),
proc AS (
    SELECT patient, count(DISTINCT code) AS distinct_procedures
    FROM silver.procedures
    GROUP BY patient
)
SELECT
    p.id                                             AS patient_id,
    p.gender,
    p.race,
    date_part('year', age(COALESCE(p.deathdate, CURRENT_DATE), p.birthdate))
                                                     AS age,
    (p.deathdate IS NOT NULL)                        AS deceased,
    COALESCE(enc.total_encounters, 0)                AS total_encounters,
    COALESCE(cond.distinct_conditions, 0)            AS distinct_conditions,
    COALESCE(proc.distinct_procedures, 0)            AS distinct_procedures,
    round(COALESCE(enc.lifetime_cost, 0), 2)         AS lifetime_cost
FROM silver.patients p
LEFT JOIN enc  ON enc.patient  = p.id
LEFT JOIN cond ON cond.patient = p.id
LEFT JOIN proc ON proc.patient = p.id;

-- 6. Revenue by month — financial view from encounters
DROP VIEW IF EXISTS gold.revenue_by_month;
CREATE VIEW gold.revenue_by_month AS
SELECT
    date_trunc('month', start)              AS month,
    count(*)                                AS encounters,
    round(sum(total_claim_cost), 2)         AS total_revenue,
    round(sum(payer_coverage), 2)           AS payer_paid,
    round(sum(total_claim_cost - payer_coverage), 2) AS patient_paid
FROM silver.encounters
WHERE start IS NOT NULL
GROUP BY month
ORDER BY month;

-- GRANTS: give haip_user read access to gold
GRANT USAGE ON SCHEMA gold TO haip_user;
GRANT SELECT ON ALL TABLES IN SCHEMA gold TO haip_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA gold GRANT SELECT ON TABLES TO haip_user;