import random
from datetime import datetime, timedelta
from typing import Dict

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

# ----------------------------
# Realism Helpers
# ----------------------------
def clamp(x, lo, hi): return max(lo, min(hi, x))

def get_bmi_range(profile):
    """Returns realistic BMI range based on profile type."""
    if profile == "Obese": return (30.0, 45.0)
    if profile == "Athlete": return (19.0, 24.5)
    if profile == "Malnourished": return (15.0, 18.0)
    return (18.5, 29.9) # Normal/General

def get_age_range(profile):
    """Returns realistic Age range based on profile type."""
    if profile == "Elderly": return (65, 95)
    if profile == "Athlete": return (18, 40)
    return (18, 80)

def calculate_bp_by_age_and_profile(age, profile, bmi):
    """
    Generates BP that realistically drifts higher with age and BMI.
    """
    base_sys = 110
    base_dia = 70
    
    # Age factor: BP tends to rise with age
    age_factor_sys = (age - 20) * 0.5 
    age_factor_dia = (age - 20) * 0.2

    # BMI factor: Higher BMI correlates with higher BP
    bmi_factor = 0
    if bmi > 25: bmi_factor = (bmi - 25) * 1.5

    # Profile modifiers
    if profile in ["Hypertensive", "Cardiac", "Diabetic", "Kidney Disease"]:
        base_sys += 15
        base_dia += 10
    
    # Random fluctuation (noise)
    noise_sys = np.random.normal(0, 8)
    noise_dia = np.random.normal(0, 5)

    sys = int(base_sys + age_factor_sys + bmi_factor + noise_sys)
    dia = int(base_dia + age_factor_dia + (bmi_factor * 0.6) + noise_dia)
    
    return clamp(sys, 90, 230), clamp(dia, 60, 140)

# ----------------------------
# 4-Level Classification Logic (Strict)
# ----------------------------
def classify_health_strict(bmi, hr, spo2, sys, dia, stress):
    """
    Classifies patient into 4 categories based on strict medical thresholds.
    """
    
    # 1. CRITICAL
    if (spo2 < 90.0) or \
       (sys >= 180 or dia >= 120) or \
       (hr > 130 or hr < 40):
        return "Critical"

    # 2. HIGH RISK
    if (spo2 < 94.0) or \
       (sys >= 140 or dia >= 90) or \
       (bmi >= 35.0) or \
       (hr > 110) or \
       (stress == 3):
        return "High Risk"

    # 3. ELEVATED RISK
    if (sys >= 120 or dia >= 80) or \
       (bmi >= 25.0) or \
       (hr > 100) or \
       (stress == 2):
        return "Elevated Risk"

    # 4. OPTIMAL
    return "Optimal"

# ----------------------------
# Noise Injection Logic (NEW)
# ----------------------------
def add_medical_noise(true_category):
    """
    Simulates real-world ambiguity and human error. 
    15% chance to mislabel a patient as a neighbor category.
    """
    # 85% chance to stay correct (Perfect Doctor/Sensor)
    if random.random() > 0.15: 
        return true_category
    
    # 15% chance to be confused with a neighbor class
    # (e.g. A doctor might say 138/88 is 'Optimal' instead of 'Elevated' depending on context)
    neighbors = {
        "Optimal": ["Elevated Risk"],
        "Elevated Risk": ["Optimal", "High Risk"],
        "High Risk": ["Elevated Risk", "Critical"],
        "Critical": ["High Risk"]
    }
    return random.choice(neighbors[true_category])

# ----------------------------
# Patient Generator
# ----------------------------
PROFILE_TYPES = [
    "Healthy", "Athlete", "Sedentary", "Obese", 
    "Diabetic", "Hypertensive", "Respiratory", "Cardiac", "Elderly"
]

def generate_patient(pid: int) -> Dict:
    profile = random.choice(PROFILE_TYPES)
    gender = random.choice(["Male", "Female"])
    
    # 1. Demographics
    age_min, age_max = get_age_range(profile)
    dob = fake.date_of_birth(minimum_age=age_min, maximum_age=age_max)
    age = int((datetime.now().date() - dob).days / 365.25)

    # 2. Height/Weight/BMI
    h_mean = 175 if gender == "Male" else 162
    height = int(np.random.normal(h_mean, 7))
    
    target_bmi_min, target_bmi_max = get_bmi_range(profile)
    target_bmi = np.random.uniform(target_bmi_min, target_bmi_max)
    weight = round(target_bmi * ((height / 100) ** 2), 1)
    bmi = round(weight / ((height / 100) ** 2), 1)

    # 3. Vitals
    if profile == "Athlete":
        hr = int(np.random.normal(55, 5)) 
    elif profile in ["Cardiac", "Obese", "Sedentary"]:
        hr = int(np.random.normal(85, 10))
    else:
        hr = int(np.random.normal(72, 8))
    
    if profile in ["Respiratory", "Cardiac"]:
        spo2 = round(np.random.choice(
            [np.random.uniform(96, 100), np.random.uniform(88, 95)], 
            p=[0.4, 0.6]
        ), 1)
    else:
        spo2 = round(np.random.uniform(96.0, 100.0), 1)

    sys, dia = calculate_bp_by_age_and_profile(age, profile, bmi)

    if profile in ["Hypertensive", "Cardiac"]:
        stress = np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])
    else:
        stress = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

    # 4. Classify (Strict -> Noisy)
    strict_cat = classify_health_strict(bmi, hr, spo2, sys, dia, stress)
    final_cat = add_medical_noise(strict_cat)

    return {
        "PatientID": pid,
        "ProfileType": profile,
        "Age": age,
        "Gender": gender,
        "Height_cm": height,
        "Weight_kg": weight,
        "BMI": bmi,
        "HeartRate_bpm": hr,
        "SpO2": spo2,
        "BP_Sys": sys,
        "BP_Dia": dia,
        "StressLevel": stress,
        "RiskCategory": final_cat # <--- The Noisy Label
    }

# ----------------------------
# Execution
# ----------------------------
def generate_dataset(n=N_PATIENTS) -> pd.DataFrame:
    patients = [generate_patient(pid) for pid in range(1, n + 1)]
    return pd.DataFrame(patients)

if __name__ == "__main__":
    df = generate_dataset(N_PATIENTS)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    print("--- Dataset Sample ---")
    print(df.head(10))
    
    print("\n--- Category Distribution ---")
    print(df["RiskCategory"].value_counts(normalize=True))
    
    # Save
    out_file = "realistic_health_data_noisy.csv"
    df.to_csv(out_file, index=False)
    print(f"\n✅ File saved: {out_file} (With 15% Label Noise)")