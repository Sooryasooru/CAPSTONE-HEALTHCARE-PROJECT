-- ============================================
-- BRONZE LAYER: raw landing tables (HAIP)
-- All TEXT; cleaning + typing happens in silver.
-- ============================================

CREATE SCHEMA IF NOT EXISTS bronze;

-- 1. Patients / Admissions (56 cols)
DROP TABLE IF EXISTS bronze.patients;
CREATE TABLE bronze.patients (
    sno TEXT, mrd_no TEXT, doa TEXT, dod TEXT, age TEXT, gender TEXT, rural TEXT,
    type_of_admission TEXT, month_year TEXT, duration_of_stay TEXT,
    duration_of_icu_stay TEXT, outcome TEXT, smoking TEXT, alcohol TEXT,
    dm TEXT, htn TEXT, cad TEXT, prior_cmp TEXT, ckd TEXT, hb TEXT, tlc TEXT,
    platelets TEXT, glucose TEXT, urea TEXT, creatinine TEXT, bnp TEXT,
    raised_cardiac_enzymes TEXT, ef TEXT, severe_anaemia TEXT, anaemia TEXT,
    stable_angina TEXT, acs TEXT, stemi TEXT, atypical_chestpain TEXT,
    heart_failure TEXT, hfref TEXT, hfnef TEXT, valvular TEXT, chb TEXT, sss TEXT,
    aki TEXT, cva_infract TEXT, cva_bleed TEXT, af TEXT, vt TEXT, psvt TEXT,
    congenital TEXT, uti TEXT, neuro_cardiogenic_syncope TEXT, orthostatic TEXT,
    infective_endocarditis TEXT, dvt TEXT, cardiogenic_shock TEXT, shock TEXT,
    pulmonary_embolism TEXT, chest_infection TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- 2. Mortality (6 cols)
DROP TABLE IF EXISTS bronze.mortality;
CREATE TABLE bronze.mortality (
    sno TEXT, mrd TEXT, age TEXT, gender TEXT, rural_urban TEXT,
    date_of_brought_dead TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- 3. Billing (10 cols)
DROP TABLE IF EXISTS bronze.billing;
CREATE TABLE bronze.billing (
    patient_id TEXT, age TEXT, gender TEXT, condition TEXT, procedure TEXT,
    cost TEXT, length_of_stay TEXT, readmission TEXT, outcome TEXT,
    satisfaction TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- 4. ICU (77 cols)
DROP TABLE IF EXISTS bronze.icu;
CREATE TABLE bronze.icu (
    subject_id TEXT, age TEXT, gender TEXT, weight_kg TEXT, height_cm TEXT,
    bmi TEXT, ethnicity TEXT, insurance TEXT, hr_mean TEXT, hr_max TEXT,
    hr_min TEXT, hr_std TEXT, sbp_mean TEXT, sbp_max TEXT, sbp_min TEXT,
    sbp_std TEXT, dbp_mean TEXT, dbp_max TEXT, dbp_min TEXT, dbp_std TEXT,
    map_mean TEXT, temp_celsius_mean TEXT, temp_celsius_max TEXT,
    temp_celsius_min TEXT, temp_celsius_std TEXT, spo2_mean TEXT, spo2_min TEXT,
    spo2_max TEXT, spo2_std TEXT, respiratory_rate_mean TEXT,
    respiratory_rate_max TEXT, respiratory_rate_min TEXT,
    respiratory_rate_std TEXT, wbc TEXT, lactate_mmol TEXT, creatinine TEXT,
    platelet_count TEXT, bilirubin_total TEXT, glucose TEXT, ph_arterial TEXT,
    pao2_fio2_ratio TEXT, inr TEXT, sodium TEXT, potassium TEXT, chloride TEXT,
    bicarbonate TEXT, hematocrit TEXT, hemoglobin TEXT, diabetes TEXT,
    hypertension TEXT, chf TEXT, copd TEXT, chronic_kidney_disease TEXT,
    liver_disease TEXT, immunosuppression TEXT, cad TEXT, atrial_fibrillation TEXT,
    cancer_active TEXT, vasopressors_flag TEXT, mechanical_ventilation TEXT,
    fio2_percent TEXT, antibiotics_24h TEXT, fluids_ml_24h TEXT,
    sedation_score TEXT, vasopressor_dose_mcg_kg_min TEXT,
    insulin_infusion_flag TEXT, sofa_score TEXT, apache_iv TEXT, qsofa TEXT,
    sirs_criteria TEXT, gcs_total TEXT, icu_los_hours TEXT,
    hospital_admit_source TEXT, icu_admit_time_hour TEXT, day_of_week TEXT,
    readmission_30day TEXT, sepsis_label TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);

-- 5. Labs (11 cols)
DROP TABLE IF EXISTS bronze.labs;
CREATE TABLE bronze.labs (
    date TEXT, test_name TEXT, result TEXT, unit TEXT, reference_range TEXT,
    status TEXT, comment TEXT, min_reference TEXT, max_reference TEXT,
    unit_description TEXT, recommended_followup TEXT,
    _loaded_at TIMESTAMP DEFAULT now()
);