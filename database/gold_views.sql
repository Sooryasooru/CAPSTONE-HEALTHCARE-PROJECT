-- GOLD LAYER: analytics-ready views (HAIP)
-- Built on silver. Each view summarizes within
-- its own domain (sources are independent).


CREATE SCHEMA IF NOT EXISTS gold;

-- 1. Admissions summary by month + admission type
DROP VIEW IF EXISTS gold.admissions_summary;
CREATE VIEW gold.admissions_summary AS
SELECT
    date_trunc('month', doa)            AS admission_month,
    type_of_admission,
    count(*)                            AS total_admissions,
    round(avg(age), 1)                  AS avg_age,
    round(avg(duration_of_stay), 1)     AS avg_stay_days,
    count(*) FILTER (WHERE outcome = 'Expiry')   AS deaths,
    count(*) FILTER (WHERE outcome = 'Discharge') AS discharges
FROM silver.patients
WHERE doa IS NOT NULL
GROUP BY admission_month, type_of_admission
ORDER BY admission_month, type_of_admission;

-- 2. Patient outcome distribution
DROP VIEW IF EXISTS gold.outcome_distribution;
CREATE VIEW gold.outcome_distribution AS
SELECT
    outcome,
    count(*)                                   AS patients,
    round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM silver.patients
WHERE outcome IS NOT NULL
GROUP BY outcome
ORDER BY patients DESC;

-- 3. Comorbidity prevalence (key risk factors)
DROP VIEW IF EXISTS gold.comorbidity_prevalence;
CREATE VIEW gold.comorbidity_prevalence AS
SELECT 'Diabetes' AS condition, sum(dm) AS cases,
       round(100.0 * sum(dm) / count(*), 1) AS prevalence_pct FROM silver.patients
UNION ALL
SELECT 'Hypertension', sum(htn), round(100.0 * sum(htn) / count(*), 1) FROM silver.patients
UNION ALL
SELECT 'CAD', sum(cad), round(100.0 * sum(cad) / count(*), 1) FROM silver.patients
UNION ALL
SELECT 'CKD', sum(ckd), round(100.0 * sum(ckd) / count(*), 1) FROM silver.patients
UNION ALL
SELECT 'Heart Failure', sum(heart_failure), round(100.0 * sum(heart_failure) / count(*), 1) FROM silver.patients
ORDER BY cases DESC;

-- 4. Billing summary by condition
DROP VIEW IF EXISTS gold.billing_by_condition;
CREATE VIEW gold.billing_by_condition AS
SELECT
    condition,
    count(*)                       AS cases,
    round(avg(cost), 2)            AS avg_cost,
    round(avg(length_of_stay), 1)  AS avg_stay,
    round(avg(satisfaction), 2)    AS avg_satisfaction,
    count(*) FILTER (WHERE readmission) AS readmissions
FROM silver.billing
GROUP BY condition
ORDER BY avg_cost DESC;

-- 5. ICU severity & outcomes summary
DROP VIEW IF EXISTS gold.icu_severity_summary;
CREATE VIEW gold.icu_severity_summary AS
SELECT
    CASE
        WHEN sofa_score < 6  THEN 'Low (0-5)'
        WHEN sofa_score < 10 THEN 'Moderate (6-9)'
        WHEN sofa_score < 15 THEN 'High (10-14)'
        ELSE 'Critical (15+)'
    END                                 AS sofa_band,
    count(*)                            AS patients,
    round(avg(age), 1)                  AS avg_age,
    round(avg(icu_los_hours), 1)        AS avg_icu_hours,
    sum(sepsis_label)                   AS sepsis_cases,
    sum(mechanical_ventilation)         AS ventilated,
    sum(readmission_30day)              AS readmissions_30d
FROM silver.icu
WHERE sofa_score IS NOT NULL
GROUP BY sofa_band
ORDER BY min(sofa_score);

-- 6. Lab results summary by test
DROP VIEW IF EXISTS gold.lab_test_summary;
CREATE VIEW gold.lab_test_summary AS
SELECT
    test_name,
    count(*)                                      AS times_run,
    count(*) FILTER (WHERE status = 'Normal')     AS normal,
    count(*) FILTER (WHERE status = 'Yüksek')     AS high
FROM silver.labs
GROUP BY test_name
ORDER BY times_run DESC;

-- GRANTS: give haip_user read access to gold
-- (re-applied on every rebuild)

GRANT USAGE ON SCHEMA gold TO haip_user;
GRANT SELECT ON ALL TABLES IN SCHEMA gold TO haip_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA gold GRANT SELECT ON TABLES TO haip_user;
