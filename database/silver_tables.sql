-- ============================================
-- SILVER LAYER: cleaned + typed tables (HAIP)
-- ============================================

CREATE SCHEMA IF NOT EXISTS silver;

-- Patients / Admissions
DROP TABLE IF EXISTS silver.patients;
CREATE TABLE silver.patients (
    sno                       INTEGER,
    mrd_no                    TEXT,
    doa                       DATE,
    dod                       DATE,
    age                       INTEGER,
    gender                    TEXT,
    rural                     TEXT,
    type_of_admission         TEXT,
    month_year                TEXT,
    duration_of_stay          INTEGER,
    duration_of_icu_stay      INTEGER,
    outcome                   TEXT,
    smoking                   INTEGER,
    alcohol                   INTEGER,
    dm                        INTEGER,
    htn                       INTEGER,
    cad                       INTEGER,
    prior_cmp                 INTEGER,
    ckd                       INTEGER,
    hb                        NUMERIC,
    tlc                       NUMERIC,
    platelets                 NUMERIC,
    glucose                   NUMERIC,
    urea                      NUMERIC,
    creatinine                NUMERIC,
    bnp                       NUMERIC,
    raised_cardiac_enzymes    INTEGER,
    ef                        NUMERIC,
    severe_anaemia            INTEGER,
    anaemia                   INTEGER,
    stable_angina             INTEGER,
    acs                       INTEGER,
    stemi                     INTEGER,
    atypical_chestpain        INTEGER,
    heart_failure             INTEGER,
    hfref                     INTEGER,
    hfnef                     INTEGER,
    valvular                  INTEGER,
    chb                       INTEGER,
    sss                       INTEGER,
    aki                       INTEGER,
    cva_infract               INTEGER,
    cva_bleed                 INTEGER,
    af                        INTEGER,
    vt                        INTEGER,
    psvt                      INTEGER,
    congenital                INTEGER,
    uti                       INTEGER,
    neuro_cardiogenic_syncope INTEGER,
    orthostatic               INTEGER,
    infective_endocarditis    INTEGER,
    dvt                       INTEGER,
    cardiogenic_shock         INTEGER,
    shock                     INTEGER,
    pulmonary_embolism        INTEGER,
    chest_infection           INTEGER,
    _loaded_at                TIMESTAMP DEFAULT now()
);

-- Mortality
DROP TABLE IF EXISTS silver.mortality;
CREATE TABLE silver.mortality (
    sno                  INTEGER,
    mrd                  TEXT,
    age                  INTEGER,
    gender               TEXT,
    rural_urban          TEXT,
    date_of_brought_dead DATE,
    _loaded_at           TIMESTAMP DEFAULT now()
);

-- Billing
DROP TABLE IF EXISTS silver.billing;
CREATE TABLE silver.billing (
    patient_id     TEXT,
    age            INTEGER,
    gender         TEXT,
    condition      TEXT,
    procedure      TEXT,
    cost           NUMERIC,
    length_of_stay INTEGER,
    readmission    BOOLEAN,
    outcome        TEXT,
    satisfaction   INTEGER,
    _loaded_at     TIMESTAMP DEFAULT now()
);

-- ICU
DROP TABLE IF EXISTS silver.icu;
CREATE TABLE silver.icu (
    subject_id                  TEXT,
    age                         INTEGER,
    gender                      TEXT,
    weight_kg                   NUMERIC,
    height_cm                   NUMERIC,
    bmi                         NUMERIC,
    ethnicity                   TEXT,
    insurance                   TEXT,
    hr_mean                     NUMERIC,
    hr_max                      NUMERIC,
    hr_min                      NUMERIC,
    hr_std                      NUMERIC,
    sbp_mean                    NUMERIC,
    sbp_max                     NUMERIC,
    sbp_min                     NUMERIC,
    sbp_std                     NUMERIC,
    dbp_mean                    NUMERIC,
    dbp_max                     NUMERIC,
    dbp_min                     NUMERIC,
    dbp_std                     NUMERIC,
    map_mean                    NUMERIC,
    temp_celsius_mean           NUMERIC,
    temp_celsius_max            NUMERIC,
    temp_celsius_min            NUMERIC,
    temp_celsius_std            NUMERIC,
    spo2_mean                   NUMERIC,
    spo2_min                    NUMERIC,
    spo2_max                    NUMERIC,
    spo2_std                    NUMERIC,
    respiratory_rate_mean       NUMERIC,
    respiratory_rate_max        NUMERIC,
    respiratory_rate_min        NUMERIC,
    respiratory_rate_std        NUMERIC,
    wbc                         NUMERIC,
    lactate_mmol                NUMERIC,
    creatinine                  NUMERIC,
    platelet_count              NUMERIC,
    bilirubin_total             NUMERIC,
    glucose                     NUMERIC,
    ph_arterial                 NUMERIC,
    pao2_fio2_ratio             NUMERIC,
    inr                         NUMERIC,
    sodium                      NUMERIC,
    potassium                   NUMERIC,
    chloride                    NUMERIC,
    bicarbonate                 NUMERIC,
    hematocrit                  NUMERIC,
    hemoglobin                  NUMERIC,
    diabetes                    INTEGER,
    hypertension                INTEGER,
    chf                         INTEGER,
    copd                        INTEGER,
    chronic_kidney_disease      INTEGER,
    liver_disease               INTEGER,
    immunosuppression           INTEGER,
    cad                         INTEGER,
    atrial_fibrillation         INTEGER,
    cancer_active               INTEGER,
    vasopressors_flag           INTEGER,
    mechanical_ventilation      INTEGER,
    fio2_percent                NUMERIC,
    antibiotics_24h             INTEGER,
    fluids_ml_24h               NUMERIC,
    sedation_score              NUMERIC,
    vasopressor_dose_mcg_kg_min NUMERIC,
    insulin_infusion_flag       INTEGER,
    sofa_score                  NUMERIC,
    apache_iv                   NUMERIC,
    qsofa                       NUMERIC,
    sirs_criteria               NUMERIC,
    gcs_total                   NUMERIC,
    icu_los_hours               NUMERIC,
    hospital_admit_source       TEXT,
    icu_admit_time_hour         INTEGER,
    day_of_week                 INTEGER,
    readmission_30day           INTEGER,
    sepsis_label                INTEGER,
    _loaded_at                  TIMESTAMP DEFAULT now()
);

-- Labs
DROP TABLE IF EXISTS silver.labs;
CREATE TABLE silver.labs (
    date                 DATE,
    test_name            TEXT,
    result               TEXT,
    unit                 TEXT,
    reference_range      TEXT,
    status               TEXT,
    comment              TEXT,
    min_reference        NUMERIC,
    max_reference        NUMERIC,
    unit_description     TEXT,
    recommended_followup TEXT,
    _loaded_at           TIMESTAMP DEFAULT now()
);

-- Grants for haip_user
GRANT USAGE ON SCHEMA silver TO haip_user;
GRANT ALL ON ALL TABLES IN SCHEMA silver TO haip_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA silver GRANT ALL ON TABLES TO haip_user;
