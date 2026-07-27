"""Generate a LARGE realistic synthetic hospital doctor dataset (~1000 rows).

~22 departments, several doctors each, realistic volume / cost / outcome
distributions, with deliberate imperfections spread through the file so the
dashboard's data-quality handling and O/E volume-suppression safeguards are
visible at scale. Synthetic, clearly labelled. Observed vs expected columns
included so risk-adjusted O/E can be computed honestly.

Run:  python make_doctors_large.py  ->  hospital_doctors_large.csv
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

DEPARTMENTS = {
    "Cardiology": 70, "General Medicine": 95, "Orthopedics": 60,
    "Neurology": 48, "Pediatrics": 55, "Oncology": 40, "Radiology": 45,
    "Emergency Medicine": 80, "Obstetrics & Gynecology": 52,
    "Gastroenterology": 38, "Pulmonology": 34, "Nephrology": 28,
    "Endocrinology": 26, "Dermatology": 30, "Psychiatry": 42,
    "Urology": 30, "Ophthalmology": 34, "ENT": 28,
    "Anesthesiology": 46, "Hematology": 22, "Rheumatology": 18,
    "Infectious Disease": 20,
}

SPECIALTY = {d: d.replace(" & ", "/") + " Specialist" for d in DEPARTMENTS}

BASE_MORT = {
    "Cardiology": 0.05, "General Medicine": 0.025, "Orthopedics": 0.012,
    "Neurology": 0.045, "Pediatrics": 0.008, "Oncology": 0.09,
    "Radiology": 0.004, "Emergency Medicine": 0.06,
    "Obstetrics & Gynecology": 0.006, "Gastroenterology": 0.02,
    "Pulmonology": 0.05, "Nephrology": 0.055, "Endocrinology": 0.015,
    "Dermatology": 0.002, "Psychiatry": 0.006, "Urology": 0.012,
    "Ophthalmology": 0.002, "ENT": 0.006, "Anesthesiology": 0.03,
    "Hematology": 0.07, "Rheumatology": 0.01, "Infectious Disease": 0.06,
}
BASE_READM = {d: min(0.22, BASE_MORT[d] * 2.6 + 0.05) for d in DEPARTMENTS}

FIRST = ["James", "Mary", "Robert", "Priya", "Wei", "Aisha", "Carlos", "Anna",
         "David", "Fatima", "John", "Sofia", "Omar", "Grace", "Liam", "Nina",
         "Raj", "Elena", "Sam", "Yuki", "Tom", "Zara", "Ivan", "Maya", "Leo",
         "Hana", "Noah", "Amara", "Kofi", "Mei", "Diego", "Lena", "Ravi",
         "Sara", "Marco", "Ada", "Felix", "Ingrid", "Juan", "Kira"]
LAST = ["Smith", "Patel", "Chen", "Khan", "Garcia", "Nguyen", "Kim", "Lopez",
        "Singh", "Rossi", "Brown", "Ali", "Hansen", "Silva", "Cohen", "Reddy",
        "Adams", "Osei", "Petrov", "Tan", "Ford", "Haddad", "Novak", "Cruz",
        "Meyer", "Bauer", "Costa", "Diaz", "Ivanov", "Sato", "Mbeki", "Park",
        "Roy", "Weber", "Yilmaz", "Zhang", "Andersen", "Bianchi", "Okafor"]


def main() -> None:
    rows = []
    did = 10000
    for dept, n_docs in DEPARTMENTS.items():
        for _ in range(n_docs):
            did += 1
            name = f"Dr. {rng.choice(FIRST)} {rng.choice(LAST)}"

            encounters = int(rng.gamma(shape=4.0, scale=140))
            encounters = max(40, min(encounters, 1600))
            patients = int(encounters * rng.uniform(0.55, 0.8))
            avg_cost = round(max(600.0, float(rng.normal(3200, 800))), 2)

            exp_mort = round(BASE_MORT[dept] * encounters, 2)
            exp_readm = round(BASE_READM[dept] * encounters, 2)
            obs_mort = int(max(0, round(exp_mort * rng.normal(1.0, 0.22))))
            obs_readm = int(max(0, round(exp_readm * rng.normal(1.0, 0.18))))

            tenure = round(float(rng.uniform(1, 30)), 1)
            join_year = 2025 - int(tenure)
            join_date = (f"{join_year}-{rng.integers(1,13):02d}-"
                         f"{rng.integers(1,28):02d}")

            rows.append({
                "doctor_id": f"D{did}", "doctor_name": name,
                "department": dept, "specialty": SPECIALTY[dept],
                "encounters": encounters, "patients_seen": patients,
                "avg_cost": avg_cost, "mortality_count": obs_mort,
                "readmission_count": obs_readm,
                "expected_mortality": exp_mort,
                "expected_readmission": exp_readm,
                "tenure_years": tenure, "join_date": join_date,
            })

    df = pd.DataFrame(rows)
    n = len(df)
    idx = np.arange(n)

    null_dept = rng.choice(idx, size=max(1, int(0.02 * n)), replace=False)
    df.loc[null_dept, "department"] = np.nan

    typos = {"Cardiology": "Cardiolgy", "Neurology": "Neurlogy",
             "Pediatrics": "Pediatrcs", "Orthopedics": "Orthopedcs"}
    for good, bad in typos.items():
        pool = df.index[df["department"] == good].tolist()
        if pool:
            hit = rng.choice(pool, size=min(3, len(pool)), replace=False)
            df.loc[hit, "department"] = bad

    low = rng.choice(idx, size=max(1, int(0.04 * n)), replace=False)
    df.loc[low, "encounters"] = rng.integers(3, 25, size=len(low))
    df.loc[low, "patients_seen"] = (df.loc[low, "encounters"] * 0.7).astype(int)

    null_out = rng.choice(idx, size=max(1, int(0.02 * n)), replace=False)
    half = len(null_out) // 2
    df.loc[null_out[:half], "mortality_count"] = np.nan
    df.loc[null_out[half:], "readmission_count"] = np.nan

    df.to_csv("hospital_doctors_large.csv", index=False)
    print(f"Wrote hospital_doctors_large.csv: {n} doctors, "
          f"{df['department'].nunique(dropna=True)} distinct labels, "
          f"{df['department'].isna().sum()} null depts")


if __name__ == "__main__":
    main()
    