import random
from datetime import datetime, timedelta, date
from typing import Dict, List

import numpy as np
import pandas as pd
from faker import Faker

# ----------------------------
# Config & Globals
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

N_PATIENTS = 10000
WEARABLE_DAYS = 14
START_DATE = datetime.now().date() - timedelta(days=WEARABLE_DAYS)

# Profiles (same distributions)
HEALTH_PROFILES = {
    "Healthy": {"wearable": {"hr_mean": 72, "hr_sd": 6, "spo2_mean": 98.5, "spo2_sd": 0.6,
                             "bp_sys_mean": 119, "bp_sys_sd": 6, "bp_dia_mean": 78, "bp_dia_sd": 4,
                             "stress_probs": (0.7, 0.25, 0.05)}},
    "Diabetic": {"wearable": {"hr_mean": 84, "hr_sd": 7, "spo2_mean": 97.0, "spo2_sd": 0.7,
                              "bp_sys_mean": 136, "bp_sys_sd": 10, "bp_dia_mean": 88, "bp_dia_sd": 7,
                              "stress_probs": (0.2, 0.6, 0.2)}},
    "Cardiac": {"wearable": {"hr_mean": 90, "hr_sd": 10, "spo2_mean": 94.5, "spo2_sd": 1.5,
                             "bp_sys_mean": 148, "bp_sys_sd": 12, "bp_dia_mean": 96, "bp_dia_sd": 8,
                             "stress_probs": (0.15, 0.5, 0.35)}},
    "Obese": {"wearable": {"hr_mean": 86, "hr_sd": 8, "spo2_mean": 96.5, "spo2_sd": 0.9,
                           "bp_sys_mean": 140, "bp_sys_sd": 10, "bp_dia_mean": 92, "bp_dia_sd": 6,
                           "stress_probs": (0.25, 0.5, 0.25)}},
    "Respiratory": {"wearable": {"hr_mean": 82, "hr_sd": 8, "spo2_mean": 93.5, "spo2_sd": 2.0,
                                 "bp_sys_mean": 132, "bp_sys_sd": 8, "bp_dia_mean": 86, "bp_dia_sd": 6,
                                 "stress_probs": (0.25, 0.55, 0.20)}},
    "Elderly": {"wearable": {"hr_mean": 80, "hr_sd": 8, "spo2_mean": 95.0, "spo2_sd": 1.2,
                             "bp_sys_mean": 142, "bp_sys_sd": 10, "bp_dia_mean": 90, "bp_dia_sd": 6,
                             "stress_probs": (0.25, 0.55, 0.20)}}
}
PROFILE_CHOICES = list(HEALTH_PROFILES.keys())

# ----------------------------
# Helpers
# ----------------------------
def clamp(x, lo, hi): return max(lo, min(hi, x))
def rnd_norm_int(mean, sd, lo=None, hi=None):
    val = int(round(np.random.normal(mean, sd)))
    return clamp(val, lo, hi) if lo is not None and hi is not None else val
def rnd_norm_float(mean, sd, lo=None, hi=None, nd=1):
    val = round(float(np.random.normal(mean, sd)), nd)
    return clamp(val, lo, hi) if lo is not None and hi is not None else val
def pick_stress(p_low, p_med, p_high): return np.random.choice([1, 2, 3], p=[p_low, p_med, p_high])

# ----------------------------
# Classification Logic
# ----------------------------
def classify_fitness(age: int, gender: str, bmi: float, hr: int, spo2: float, bp_sys: int, bp_dia: int, stress: int) -> str:
    """
    Classifies a patient into Fit / Slightly Unfit / Severe based on
    age, gender, and physiological parameters.
    """

    gender = gender.capitalize()
    # Determine age group
    if age < 40:
        group = "18-39"
    elif age < 60:
        group = "40-59"
    else:
        group = "60+"

    # Normal reference values
    norms = {
        ("18-39", "Male"):  {"bmi": (18.5, 24.9), "hr": (70, 72), "spo2": (95, 100), "bp": (119, 70), "stress": [2]},
        ("18-39", "Female"):{"bmi": (18.5, 24.9), "hr": (78, 82), "spo2": (95, 100), "bp": (110, 68), "stress": [2, 3]},
        ("40-59", "Male"):  {"bmi": (18.5, 24.9), "hr": (69, 73), "spo2": (95, 100), "bp": (124, 77), "stress": [2]},
        ("40-59", "Female"):{"bmi": (18.5, 24.9), "hr": (76, 80), "spo2": (95, 100), "bp": (122, 74), "stress": [2]},
        ("60+", "Male"):    {"bmi": (22, 26), "hr": (66, 86), "spo2": (93, 100), "bp": (133, 69), "stress": [1, 2]},
        ("60+", "Female"):  {"bmi": (22, 26), "hr": (66, 90), "spo2": (93, 100), "bp": (139, 68), "stress": [1, 2]},
    }

    ref = norms[(group, gender)]

    # small tolerance for biological variance
    tol_bmi = 1.0
    tol_hr = 5
    tol_bp = 10
    tol_spo2 = 2

    score = 0

    # --- BMI ---
    if ref["bmi"][0] - tol_bmi <= bmi <= ref["bmi"][1] + tol_bmi:
        score += 0
    elif ref["bmi"][0] - 3 <= bmi <= ref["bmi"][1] + 3:
        score += 1
    else:
        score += 2

    # --- Heart Rate ---
    if ref["hr"][0] - tol_hr <= hr <= ref["hr"][1] + tol_hr:
        score += 0
    elif ref["hr"][0] - 10 <= hr <= ref["hr"][1] + 10:
        score += 1
    else:
        score += 2

    # --- SpO2 ---
    if spo2 >= ref["spo2"][0] - tol_spo2:
        score += 0
    elif spo2 >= ref["spo2"][0] - 4:
        score += 1
    else:
        score += 2

    # --- Blood Pressure ---
    sys_ref, dia_ref = ref["bp"]
    if (sys_ref - tol_bp <= bp_sys <= sys_ref + tol_bp) and (dia_ref - 5 <= bp_dia <= dia_ref + 5):
        score += 0
    elif (sys_ref - 15 <= bp_sys <= sys_ref + 15) or (dia_ref - 10 <= bp_dia <= dia_ref + 10):
        score += 1
    else:
        score += 2

    # --- Stress Level ---
    if stress in ref["stress"]:
        score += 0
    elif stress == 3:
        score += 1
    else:
        score += 2

    # --- Combine score ---
    if score <= 2:
        return "Fit"
    elif score <= 5:
        return "Slightly Unfit"
    else:
        return "Severe"

# ----------------------------
# Patient generator
# ----------------------------
def generate_patient(pid: int) -> Dict:
    profile = random.choice(PROFILE_CHOICES)
    gender = random.choice(["Male", "Female"])
    dob = fake.date_of_birth(minimum_age=20, maximum_age=90)
    age = int((datetime.now().date() - dob).days / 365.25)
    height = rnd_norm_int(170, 9, 150, 195)
    weight = rnd_norm_int(72, 14, 45, 140)
    bmi = round(weight / ((height / 100) ** 2), 1)

    wcfg = HEALTH_PROFILES[profile]["wearable"]
    hr = rnd_norm_int(wcfg["hr_mean"], wcfg["hr_sd"], 45, 140)
    spo2 = rnd_norm_float(wcfg["spo2_mean"], wcfg["spo2_sd"], 85, 100, nd=1)
    sys = rnd_norm_int(wcfg["bp_sys_mean"], wcfg["bp_sys_sd"], 90, 220)
    dia = rnd_norm_int(wcfg["bp_dia_mean"], wcfg["bp_dia_sd"], 50, 140)
    stress = pick_stress(*wcfg["stress_probs"])

    fitness_category = classify_fitness(age, gender, bmi, hr, spo2, sys, dia, stress)

    return {
        "PatientID": pid,
        "DOB": dob,
        "Age": age,
        "Gender": gender,
        "HeartRate_bpm": hr,
        "SpO2": spo2,
        "StressLevel": stress,
        "BP_Sys": sys,
        "BP_Dia": dia,
        "BMI": bmi,
        "FitnessCategory": fitness_category
    }

# ----------------------------
# Dataset generator
# ----------------------------
def generate_dataset(n_patients=N_PATIENTS) -> pd.DataFrame:
    patients = [generate_patient(pid) for pid in range(1, n_patients + 1)]
    df = pd.DataFrame(patients)
    df["DOB"] = pd.to_datetime(df["DOB"])
    return df

# ----------------------------
# Export
# ----------------------------
def export_csv(df: pd.DataFrame, out_path: str = "./single_summary.csv"):
    df.to_csv(out_path, index=False)
    print(f"✅ File written: {out_path} ({len(df)} records)")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    df = generate_dataset(N_PATIENTS)
    print(df.head())
    export_csv(df)
