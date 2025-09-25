# synth_health_data.py
import random
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple
import os

import numpy as np
import pandas as pd
from faker import Faker
from sqlalchemy import create_engine, text

# ----------------------------
# Config & Globals
# ----------------------------
import time
SEED = int(time.time()) % 100000
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_PATIENTS = 100          # Default number of patients per batch
WEARABLE_DAYS = 14
ENCOUNTERS_RANGE = (1, 3)
START_DATE = datetime.now().date() - timedelta(days=WEARABLE_DAYS)
PROFILE_CHOICES = ["Healthy", "Diabetic", "Cardiac", "Obese", "Respiratory", "Elderly"]

# ----------------------------
# HEALTH_PROFILES dict
# ----------------------------
# Keep your HEALTH_PROFILES dict here exactly as in your previous code
# For brevity, it's assumed to be defined as before with all 6 profiles.

HEALTH_PROFILES = {
    "Healthy": {
        "conditions": [],
        "wearable": {
            "sleep_mean": 7.6, "sleep_sd": 0.6,
            "steps_mean": 11000, "steps_sd": 1800,
            "hr_mean": 72, "hr_sd": 6,
            "spo2_mean": 98.5, "spo2_sd": 0.6,
            "bp_sys_mean": 119, "bp_sys_sd": 6,
            "bp_dia_mean": 78, "bp_dia_sd": 4,
            "stress_probs": (0.7, 0.25, 0.05)  # Low, Med, High
        },
        "labs": {
            "chol_total_mean": 180, "chol_total_sd": 20,     # mg/dL
            "ldl_mean": 100, "ldl_sd": 15,
            "hba1c_mean": 5.2, "hba1c_sd": 0.2,              # %
            "fpg_mean": 92, "fpg_sd": 8                      # fasting plasma glucose mg/dL
        }
    },
    "Diabetic": {
        "conditions": ["Diabetes"],
        "wearable": {
            "sleep_mean": 6.5, "sleep_sd": 0.7,
            "steps_mean": 5800, "steps_sd": 1500,
            "hr_mean": 84, "hr_sd": 7,
            "spo2_mean": 97.0, "spo2_sd": 0.7,
            "bp_sys_mean": 136, "bp_sys_sd": 10,
            "bp_dia_mean": 88, "bp_dia_sd": 7,
            "stress_probs": (0.2, 0.6, 0.2)
        },
        "labs": {
            "chol_total_mean": 205, "chol_total_sd": 25,
            "ldl_mean": 130, "ldl_sd": 20,
            "hba1c_mean": 8.2, "hba1c_sd": 1.0,   # clearly abnormal on average
            "fpg_mean": 160, "fpg_sd": 25
        }
    },
    "Cardiac": {
        "conditions": ["Heart Disease", "Hypertension"],
        "wearable": {
            "sleep_mean": 6.1, "sleep_sd": 0.8,
            "steps_mean": 5200, "steps_sd": 1500,
            "hr_mean": 90, "hr_sd": 10,
            "spo2_mean": 94.5, "spo2_sd": 1.5,
            "bp_sys_mean": 148, "bp_sys_sd": 12,
            "bp_dia_mean": 96, "bp_dia_sd": 8,
            "stress_probs": (0.15, 0.5, 0.35)
        },
        "labs": {
            "chol_total_mean": 250, "chol_total_sd": 30,
            "ldl_mean": 165, "ldl_sd": 25,
            "hba1c_mean": 5.8, "hba1c_sd": 0.4,
            "fpg_mean": 105, "fpg_sd": 12
        }
    },
    "Obese": {
        "conditions": ["Obesity", "Hypertension"],
        "wearable": {
            "sleep_mean": 5.9, "sleep_sd": 0.7,
            "steps_mean": 4200, "steps_sd": 1200,
            "hr_mean": 86, "hr_sd": 8,
            "spo2_mean": 96.5, "spo2_sd": 0.9,
            "bp_sys_mean": 140, "bp_sys_sd": 10,
            "bp_dia_mean": 92, "bp_dia_sd": 6,
            "stress_probs": (0.25, 0.5, 0.25)
        },
        "labs": {
            "chol_total_mean": 220, "chol_total_sd": 25,
            "ldl_mean": 140, "ldl_sd": 20,
            "hba1c_mean": 6.1, "hba1c_sd": 0.6,  # often prediabetes
            "fpg_mean": 112, "fpg_sd": 15
        }
    },
    "Respiratory": {
        "conditions": ["Asthma"],
        "wearable": {
            "sleep_mean": 6.2, "sleep_sd": 0.8,
            "steps_mean": 6000, "steps_sd": 1500,
            "hr_mean": 82, "hr_sd": 8,
            "spo2_mean": 93.5, "spo2_sd": 2.0,   # lower SpO2 typical
            "bp_sys_mean": 132, "bp_sys_sd": 8,
            "bp_dia_mean": 86, "bp_dia_sd": 6,
            "stress_probs": (0.25, 0.55, 0.20)
        },
        "labs": {
            "chol_total_mean": 195, "chol_total_sd": 25,
            "ldl_mean": 115, "ldl_sd": 20,
            "hba1c_mean": 5.6, "hba1c_sd": 0.3,
            "fpg_mean": 98, "fpg_sd": 10
        }
    },
    "Elderly": {
        "conditions": ["Hypertension", "Arthritis"],
        "wearable": {
            "sleep_mean": 6.1, "sleep_sd": 0.8,
            "steps_mean": 3800, "steps_sd": 1100,
            "hr_mean": 80, "hr_sd": 8,
            "spo2_mean": 95.0, "spo2_sd": 1.2,
            "bp_sys_mean": 142, "bp_sys_sd": 10,
            "bp_dia_mean": 90, "bp_dia_sd": 6,
            "stress_probs": (0.25, 0.55, 0.20)
        },
        "labs": {
            "chol_total_mean": 210, "chol_total_sd": 25,
            "ldl_mean": 130, "ldl_sd": 20,
            "hba1c_mean": 5.9, "hba1c_sd": 0.4,
            "fpg_mean": 108, "fpg_sd": 12
        }
    }
}

# ----------------------------
# Helper functions
# ----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def rnd_norm_int(mean, sd, lo=None, hi=None):
    val = int(round(np.random.normal(mean, sd)))
    if lo is not None and hi is not None:
        val = clamp(val, lo, hi)
    return val

def rnd_norm_float(mean, sd, lo=None, hi=None, nd=1):
    val = round(float(np.random.normal(mean, sd)), nd)
    if lo is not None and hi is not None:
        val = clamp(val, lo, hi)
    return val

def pick_stress(p_low, p_med, p_high):
    return np.random.choice([1,2,3], p=[p_low, p_med, p_high])

# ----------------------------
# Patient generation
# ----------------------------
def generate_patient(pid: int) -> Dict:
    profile = random.choice(PROFILE_CHOICES)
    gender = random.choice(["Male", "Female"])
    dob = fake.date_of_birth(minimum_age=20, maximum_age=90)
    height = rnd_norm_int(170, 9, 150, 195)
    weight = rnd_norm_int(72, 14, 45, 140)
    bmi = round(weight / ((height / 100) ** 2), 1)
    name = fake.name_male() if gender == "Male" else fake.name_female()
    phone = str(random.randint(6000000000, 9999999999))
    return {
        "PatientID": pid,
        "Name": name,
        "DOB": dob,
        "Gender": gender,
        "Height_cm": height,
        "Weight_kg": weight,
        "BMI": bmi,
        "Address": fake.address().replace("\n", ", "),
        "Phone": phone,
        "Profile": profile
    }

def generate_conditions(pid: int, profile: str) -> List[Dict]:
    rows = []
    year_now = datetime.now().year
    for cond in HEALTH_PROFILES[profile]["conditions"]:
        year_dx = random.randint(year_now - 20, year_now - 1)
        status = random.choices(["Ongoing", "Recovered"], weights=[0.75, 0.25])[0]
        rows.append({
            "PatientID": pid,
            "Condition": cond,
            "YearDiagnosed": year_dx,
            "Status": status
        })
    return rows

# ----------------------------
# Wearable generation
# ----------------------------
def generate_wearables(pid: int, profile: str, days: int = WEARABLE_DAYS, start_date: date = START_DATE) -> List[Dict]:
    wcfg = HEALTH_PROFILES[profile]["wearable"]
    rows = []
    for i in range(days):
        d = start_date + timedelta(days=i)
        sleep = rnd_norm_float(wcfg["sleep_mean"], wcfg["sleep_sd"], 3.0, 10.0, nd=1)
        steps = rnd_norm_int(wcfg["steps_mean"], wcfg["steps_sd"], 0, 30000)
        hr = rnd_norm_int(wcfg["hr_mean"], wcfg["hr_sd"], 45, 140)
        spo2 = rnd_norm_float(wcfg["spo2_mean"], wcfg["spo2_sd"], 85.0, 100.0, nd=1)
        sys = rnd_norm_int(wcfg["bp_sys_mean"], wcfg["bp_sys_sd"], 90, 220)
        dia = rnd_norm_int(wcfg["bp_dia_mean"], wcfg["bp_dia_sd"], 50, 140)
        stress = pick_stress(*wcfg["stress_probs"])
        rows.append({
            "PatientID": pid,
            "Timestamp": datetime.combine(d, datetime.min.time()) + timedelta(hours=random.randint(6, 22)),
            "Sleep_hrs": sleep,
            "Steps": steps,
            "HeartRate_bpm": hr,
            "SpO2": spo2,
            "StressLevel": stress,
            "BP_Sys": sys,
            "BP_Dia": dia
        })
    return rows

# ----------------------------
# Encounter Notes & Labs/Medications
# ----------------------------
# Keep your previous encounter_note and generate_encounters_and_orders functions here exactly

def encounter_note(profile: str, labs_summary: str) -> str:
    base = {
        "Healthy": "Routine health check. ",
        "Diabetic": "Diabetes follow-up. ",
        "Cardiac": "Cardiovascular review. ",
        "Obese": "Weight and blood pressure management. ",
        "Respiratory": "Asthma/respiratory symptom review. ",
        "Elderly": "Geriatric follow-up. "
    }[profile]
    return base + labs_summary

def generate_encounters_and_orders(pid: int, profile: str, start_encounter_id: int, n_encounters: int
                                   ) -> Tuple[List[Dict], List[Dict], List[Dict], int]:
    ecfg = HEALTH_PROFILES[profile]["labs"]
    encounters, meds, labs = [], [], []
    next_eid = start_encounter_id

    for _ in range(n_encounters):
        eid = next_eid
        next_eid += 1

        # Encounter date within last 1–18 months
        d = datetime.now().date() - timedelta(days=random.randint(30, 540))

        # Profile-driven labs (with some probability of control)
        rows_labs = []
        rows_meds = []
        lab_summ_parts = []

        if profile == "Diabetic":
            # HbA1c & FPG – usually abnormal, but 25% chance controlled
            controlled = random.random() < 0.25
            hba1c = rnd_norm_float(ecfg["hba1c_mean"], ecfg["hba1c_sd"], 5.5, 12.5, nd=1)
            fpg = rnd_norm_int(ecfg["fpg_mean"], ecfg["fpg_sd"], 80, 300)
            if controlled:
                hba1c = rnd_norm_float(6.2, 0.3, 5.6, 6.8, nd=1)
                fpg = rnd_norm_int(115, 10, 95, 135)

            rows_labs += [
                {"EncounterID": eid, "TestName": "HbA1c", "Result": f"{hba1c}", "Unit": "%", "NormalRange": "<5.7"},
                {"EncounterID": eid, "TestName": "Fasting Glucose", "Result": f"{fpg}", "Unit": "mg/dL", "NormalRange": "70-99"}
            ]
            rows_meds += [
                {"EncounterID": eid, "DrugName": "Metformin", "Dosage": "500mg", "Frequency": "2/day", "Duration": "6 months"}
            ]
            if not controlled and random.random() < 0.35:
                rows_meds.append({"EncounterID": eid, "DrugName": "Insulin (basal)", "Dosage": "10U", "Frequency": "1/day", "Duration": "ongoing"})
            lab_summ_parts.append(f"HbA1c={hba1c}%, FPG={fpg} mg/dL.")

        elif profile == "Cardiac":
            chol = rnd_norm_int(ecfg["chol_total_mean"], ecfg["chol_total_sd"], 150, 350)
            ldl = rnd_norm_int(ecfg["ldl_mean"], ecfg["ldl_sd"], 70, 260)
            rows_labs += [
                {"EncounterID": eid, "TestName": "Total Cholesterol", "Result": f"{chol}", "Unit": "mg/dL", "NormalRange": "<200"},
                {"EncounterID": eid, "TestName": "LDL-C", "Result": f"{ldl}", "Unit": "mg/dL", "NormalRange": "<100"}
            ]
            rows_meds += [
                {"EncounterID": eid, "DrugName": "Atorvastatin", "Dosage": "10mg", "Frequency": "1/day", "Duration": "6 months"}
            ]
            if random.random() < 0.3:
                rows_meds.append({"EncounterID": eid, "DrugName": "ACE inhibitor", "Dosage": "5mg", "Frequency": "1/day", "Duration": "6 months"})
            lab_summ_parts.append(f"Chol={chol}, LDL={ldl} mg/dL.")

        elif profile == "Obese":
            bp_sys = rnd_norm_int(HEALTH_PROFILES["Obese"]["wearable"]["bp_sys_mean"], 8, 120, 180)
            bp_dia = rnd_norm_int(HEALTH_PROFILES["Obese"]["wearable"]["bp_dia_mean"], 6, 70, 110)
            chol = rnd_norm_int(ecfg["chol_total_mean"], ecfg["chol_total_sd"], 160, 300)
            rows_labs += [
                {"EncounterID": eid, "TestName": "Blood Pressure", "Result": f"{bp_sys}/{bp_dia}", "Unit": "mmHg", "NormalRange": "120/80"},
                {"EncounterID": eid, "TestName": "Total Cholesterol", "Result": f"{chol}", "Unit": "mg/dL", "NormalRange": "<200"}
            ]
            rows_meds += [{"EncounterID": eid, "DrugName": "Amlodipine", "Dosage": "5mg", "Frequency": "1/day", "Duration": "6 months"}]
            if random.random() < 0.25:
                rows_meds.append({"EncounterID": eid, "DrugName": "Weight-loss counseling", "Dosage": "-", "Frequency": "weekly", "Duration": "3 months"})
            lab_summ_parts.append(f"BP={bp_sys}/{bp_dia} mmHg, Chol={chol}.")

        elif profile == "Respiratory":
            fev1 = rnd_norm_int(68, 8, 40, 95)  # % predicted
            rows_labs += [{"EncounterID": eid, "TestName": "Spirometry (FEV1 %)", "Result": f"{fev1}", "Unit": "%", "NormalRange": ">80"}]
            rows_meds += [{"EncounterID": eid, "DrugName": "Inhaled corticosteroid", "Dosage": "2 puffs", "Frequency": "bid", "Duration": "ongoing"}]
            lab_summ_parts.append(f"FEV1={fev1}% predicted.")

        elif profile == "Elderly":
            bp_sys = rnd_norm_int(HEALTH_PROFILES["Elderly"]["wearable"]["bp_sys_mean"], 10, 120, 180)
            bp_dia = rnd_norm_int(HEALTH_PROFILES["Elderly"]["wearable"]["bp_dia_mean"], 8, 70, 110)
            rows_labs += [{"EncounterID": eid, "TestName": "Blood Pressure", "Result": f"{bp_sys}/{bp_dia}", "Unit": "mmHg", "NormalRange": "120/80"}]
            if random.random() < 0.4:
                rows_meds.append({"EncounterID": eid, "DrugName": "Thiazide diuretic", "Dosage": "12.5mg", "Frequency": "1/day", "Duration": "6 months"})
            lab_summ_parts.append(f"BP={bp_sys}/{bp_dia} mmHg.")

        else:  # Healthy
            chol = rnd_norm_int(HEALTH_PROFILES["Healthy"]["labs"]["chol_total_mean"],
                                HEALTH_PROFILES["Healthy"]["labs"]["chol_total_sd"], 140, 220)
            hba1c = rnd_norm_float(HEALTH_PROFILES["Healthy"]["labs"]["hba1c_mean"],
                                   HEALTH_PROFILES["Healthy"]["labs"]["hba1c_sd"], 4.8, 5.8, nd=1)
            rows_labs += [
                {"EncounterID": eid, "TestName": "Total Cholesterol", "Result": f"{chol}", "Unit": "mg/dL", "NormalRange": "<200"},
                {"EncounterID": eid, "TestName": "HbA1c", "Result": f"{hba1c}", "Unit": "%", "NormalRange": "<5.7"}
            ]
            # Usually no meds
            if random.random() < 0.1:
                rows_meds.append({"EncounterID": eid, "DrugName": "Multivitamin", "Dosage": "1 tab", "Frequency": "1/day", "Duration": "1 month"})
            lab_summ_parts.append(f"Chol={chol} mg/dL, HbA1c={hba1c}%.")

        labs_summary = " ".join(lab_summ_parts)
        encounters.append({
            "EncounterID": eid,
            "PatientID": pid,
            "Date": d,
            "Notes": encounter_note(profile, labs_summary)
        })
        meds.extend(rows_meds)
        labs.extend(rows_labs)

    return encounters, meds, labs, next_eid


# ----------------------------
# Dataset generator
# ----------------------------
def generate_dataset(n_patients=N_PATIENTS, start_pid=1, start_eid=1) -> Dict[str, pd.DataFrame]:
    patients, wearable, conditions = [], [], []
    encounters, medications, labtests = [], [], []

    next_encounter_id = start_eid
    for i in range(n_patients):
        pid = start_pid + i
        p = generate_patient(pid)
        patients.append(p)

        conditions.extend(generate_conditions(pid, p["Profile"]))
        wearable.extend(generate_wearables(pid, p["Profile"], WEARABLE_DAYS, START_DATE))

        n_enc = random.randint(*ENCOUNTERS_RANGE)
        encs, meds, labs, next_encounter_id = generate_encounters_and_orders(
            pid, p["Profile"], next_encounter_id, n_enc
        )
        encounters.extend(encs)
        medications.extend(meds)
        labtests.extend(labs)

    # Build DataFrames
    df_patients = pd.DataFrame(patients)
    df_encounters = pd.DataFrame(encounters)
    df_wearable = pd.DataFrame(wearable)
    df_meds = pd.DataFrame(medications)
    df_labs = pd.DataFrame(labtests)
    df_conditions = pd.DataFrame(conditions)

    # Ensure dtypes
    df_patients["PatientID"] = df_patients["PatientID"].astype(int)
    df_encounters["EncounterID"] = df_encounters["EncounterID"].astype(int)
    df_encounters["PatientID"] = df_encounters["PatientID"].astype(int)
    df_wearable["PatientID"] = df_wearable["PatientID"].astype(int)
    df_wearable["Timestamp"] = pd.to_datetime(df_wearable["Timestamp"])
    df_wearable["StressLevel"] = df_wearable["StressLevel"].astype(int)
    df_conditions["PatientID"] = df_conditions["PatientID"].astype(int)
    df_conditions["YearDiagnosed"] = df_conditions["YearDiagnosed"].astype(int)

    return {
        "Patients": df_patients,
        "Encounters": df_encounters,
        "WearableData": df_wearable,
        "Medications": df_meds,
        "LabTests": df_labs,
        "Conditions": df_conditions
    }

# ----------------------------
# Auto Append CSV + MySQL
# ----------------------------
def export_csv_and_mysql_auto(out_dir: str = "./exports", sql_uri: str = None, n_new_patients: int = 100):
    os.makedirs(out_dir, exist_ok=True)

    # --- Detect max IDs from existing CSVs ---
    def max_id(file_path, col):
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df[col].max()
        return 0

    max_pid_csv = max_id(f"{out_dir}/Patients.csv", "PatientID")
    max_eid_csv = max_id(f"{out_dir}/Encounters.csv", "EncounterID")

    # --- Detect max IDs from SQL (optional) ---
    max_pid_sql, max_eid_sql = 0, 0
    engine = None
    if sql_uri:
        engine = create_engine(sql_uri)
        try:
            with engine.connect() as conn:
                for table, col in [("Patients", "PatientID"), ("Encounters", "EncounterID")]:
                    res = conn.execute(text(f"SELECT MAX({col}) FROM {table}")).scalar()
                    if table == "Patients": max_pid_sql = res or 0
                    if table == "Encounters": max_eid_sql = res or 0
        except Exception as e:
            print(f"SQL connection failed or tables don't exist yet: {e}")

    # --- Determine starting IDs ---
    start_pid = max(max_pid_csv, max_pid_sql) + 1
    start_eid = max(max_eid_csv, max_eid_sql) + 1

    # --- Generate new batch ---
    new_data = generate_dataset(n_new_patients, start_pid=start_pid, start_eid=start_eid)

    # --- Append to CSVs ---
    for name, df in new_data.items():
        file_path = f"{out_dir}/{name}.csv"
        df.to_csv(file_path, mode="a", header=not os.path.exists(file_path), index=False)
        print(f"Appended {len(df)} rows -> {file_path}")

    # --- Append to SQL ---
    if engine:
        for name, df in new_data.items():
            df.to_sql(name, con=engine, if_exists="append", index=False)
            print(f"Appended {len(df)} rows -> SQL table {name}")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    sql_uri = "mysql+pymysql://root:sql_my1country@localhost:3306/BTP"
    export_csv_and_mysql_auto("./exports", sql_uri=sql_uri, n_new_patients=100)
