# src/agent3.py
from typing import Dict, List
from langchain.tools import tool
from sqlalchemy import create_engine, text
import pandas as pd
import re
from datetime import datetime, timedelta
import os

# === DB Connection (Using Environment Variable) ===
DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
engine = create_engine(DB_URI)

# === Utilities ===
def _clean_patient_ids(patient_ids_str: str) -> List[int]:
    """Extract a list of numeric patient IDs from a comma-separated string."""
    if not isinstance(patient_ids_str, str):
        return []
    # Use regex to find all numbers in the string
    ids = re.findall(r'\d+', patient_ids_str)
    return [int(id) for id in ids]

def _normalize_wearable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various schema names to a consistent internal schema."""
    df = df.copy()
    aliases = {
        "Steps": ["Steps", "steps"],
        "HeartRate": ["HeartRate", "HeartRate_bpm", "heart_rate"],
        "OxygenLevel": ["OxygenLevel", "SpO2", "oxygen_level"],
        "StressLevel": ["StressLevel", "stress_level"],
        "SystolicBP": ["SystolicBP", "BP_Sys", "bp_sys"],
        "DiastolicBP": ["DiastolicBP", "BP_Dia", "bp_dia"],
        "SleepHours": ["SleepHours", "Sleep_hrs", "sleep_hours", "sleep_cycle"],
        "Timestamp": ["Timestamp", "ts", "Date", "date"]
    }
    for std, opts in aliases.items():
        for o in opts:
            if o in df.columns:
                df.rename(columns={o: std}, inplace=True)
                break
    for col in ["Steps", "HeartRate", "OxygenLevel", "StressLevel", "SystolicBP", "DiastolicBP", "SleepHours"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

# === Risk Scoring Logic ===
def calculate_risk_scores(patient_row: pd.Series, wearable_row: pd.Series):
    """Calculate health risks based on patient and wearable data."""
    risks: Dict[str, float] = {}
    explanations: List[str] = []

    # Hypertension
    sys_, dia_ = wearable_row.get("SystolicBP"), wearable_row.get("DiastolicBP")
    if pd.notna(sys_) and pd.notna(dia_) and (sys_ > 140 or dia_ > 90):
        risks["hypertension"] = 1.0
        explanations.append(f"High BP detected ({int(sys_)}/{int(dia_)} mmHg).")
    else:
        risks["hypertension"] = 0.2
        if pd.notna(sys_) and pd.notna(dia_):
            explanations.append(f"Blood pressure normal ({int(sys_)}/{int(dia_)} mmHg).")
        else:
            explanations.append("Blood pressure data not available.")

    # Diabetes (BMI proxy)
    bmi = patient_row.get("BMI")
    if pd.notna(bmi) and bmi > 30:
        risks["diabetes"] = 0.8
        explanations.append(f"High BMI ({bmi}), possible diabetes risk.")
    else:
        risks["diabetes"] = 0.2
        if pd.notna(bmi):
            explanations.append(f"BMI ({bmi}) is normal.")
        else:
            explanations.append("BMI not available.")

    # Cardiac (tachycardia proxy)
    hr = wearable_row.get("HeartRate")
    if pd.notna(hr) and hr > 110:
        risks["cardiac"] = 0.8
        explanations.append(f"Elevated heart rate ({int(hr)} bpm).")
    else:
        risks["cardiac"] = 0.2
        if pd.notna(hr):
            explanations.append(f"Heart rate ({int(hr)} bpm) within normal range.")
        else:
            explanations.append("Heart rate not available.")

    # Respiratory (SpO2 proxy)
    o2 = wearable_row.get("OxygenLevel")
    if pd.notna(o2) and o2 < 92:
        risks["respiratory"] = 0.8
        explanations.append(f"Low oxygen level ({o2}%).")
    else:
        risks["respiratory"] = 0.2
        if pd.notna(o2):
            explanations.append(f"Oxygen level ({o2}%) is normal.")
        else:
            explanations.append("Oxygen level not available.")

    return risks, explanations

def explain_anomaly(row: pd.Series) -> str:
    """Generate a human-readable reason for a detected anomaly."""
    reasons = []
    if pd.notna(row.get("HeartRate")) and row["HeartRate"] > 120:
        reasons.append(f"Unusually high HR {int(row['HeartRate'])} bpm")
    if pd.notna(row.get("OxygenLevel")) and row["OxygenLevel"] < 90:
        reasons.append(f"Low oxygen {row['OxygenLevel']}%")
    if pd.notna(row.get("Steps")) and pd.notna(row.get("SleepHours")):
        if row["Steps"] < 500 and row["SleepHours"] < 4:
            reasons.append("Low activity & poor sleep")
    return ", ".join(reasons) if reasons else "No clear anomaly detected"


# === Tools ===
@tool("summarize_patient_conditions", return_direct=True)
def summarize_patient_conditions_tool(condition: str) -> str:
    """
    Counts how many patients meet a specific condition.
    Valid conditions are: 'diabetes', 'hypertension', 'obese' (BMI > 30), 'senior' (Age > 65).
    """
    condition = condition.lower().strip("'\" ")

    sql_query = ""
    # This logic now maps to the correct tables and columns from your new schema.
    if condition == 'diabetes':
        # Queries the new Conditions table directly
        sql_query = "SELECT COUNT(*) FROM Conditions WHERE `Condition` LIKE '%diabetes%'"
    elif condition == 'hypertension':
        # Queries the new Conditions table directly
        sql_query = "SELECT COUNT(*) FROM Conditions WHERE `Condition` LIKE '%hypertension%'"
    elif condition == 'obese':
        # Correctly uses the BMI column from the Patients table
        sql_query = "SELECT COUNT(*) FROM Patients WHERE BMI > 30"
    elif condition == 'senior':
        # Calculates age directly from the DOB column in the database
        # TIMESTAMPDIFF(YEAR, DOB, CURDATE()) is a standard SQL function for age calculation
        sql_query = "SELECT COUNT(*) FROM Patients WHERE TIMESTAMPDIFF(YEAR, DOB, CURDATE()) > 65"
    else:
        valid_conditions = ['diabetes', 'hypertension', 'obese', 'senior']
        return f"Sorry, I cannot query for '{condition}'. Valid conditions are: {', '.join(valid_conditions)}"

    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql_query)).scalar_one()
        return f"Found {result} patients matching the condition: '{condition}'."
    except Exception as e:
        return f"Error querying the database for condition '{condition}': {e}"
    
@tool("risk_score", return_direct=True)
def risk_score_tool(PatientIDs: str) -> str:
    """
    Calculate patient risk scores from the *latest* wearable records.
    Accepts a single PatientID or a comma-separated list of PatientIDs.
    """
    try:
        pids = _clean_patient_ids(PatientIDs)
        if not pids:
            return "Please provide at least one valid PatientID."

        # Fetch all patient and wearable data in one go
        patients = pd.read_sql(text("SELECT * FROM Patients WHERE PatientID IN :pids"), engine, params={"pids": pids})
        wearables = pd.read_sql(
            text("""
                SELECT w.* FROM WearableData w
                INNER JOIN (
                    SELECT PatientID, MAX(Timestamp) AS MaxTS
                    FROM WearableData
                    WHERE PatientID IN :pids
                    GROUP BY PatientID
                ) m ON w.PatientID = m.PatientID AND w.Timestamp = m.MaxTS
            """),
            engine,
            params={"pids": pids},
        )
        wearables = _normalize_wearable_columns(wearables)

        results = []
        for pid in pids:
            p_row = patients[patients["PatientID"] == pid]
            w_row = wearables[wearables["PatientID"] == pid]

            if p_row.empty:
                results.append(f"--- Patient ID {pid}: NOT FOUND ---")
                continue
            if w_row.empty:
                name = p_row.iloc[0].get("Name", f"Patient {pid}")
                results.append(f"--- Risk scores for {name} (ID {pid}): No wearable data found. ---")
                continue

            p = p_row.iloc[0]
            w = w_row.iloc[0]
            risks, explanations = calculate_risk_scores(p, w)
            name = p.get("Name", f"Patient {pid}")
            result_str = (
                f"--- Risk scores for {name} (ID {pid}) based on latest wearable ({w.get('Timestamp', 'n/a')}) ---\n"
                f"{risks}\n"
                f"Reasons: " + "; ".join(explanations)
            )
            results.append(result_str)

        return "\n\n".join(results)
    except Exception as e:
        return f"Error: {e}"

@tool("detect_anomalies", return_direct=True)
def detect_anomalies_tool(PatientIDs: str) -> str:
    """
    Detect anomalies in wearable time series using IsolationForest.
    Accepts a single PatientID or a comma-separated list of PatientIDs.
    Analyzes data from the last 90 days.
    """
    try:
        pids = _clean_patient_ids(PatientIDs)
        if not pids:
            return "Please provide at least one valid PatientID."

        start_date = datetime.utcnow() - timedelta(days=90)
        wearables = pd.read_sql(
            text("SELECT * FROM WearableData WHERE PatientID IN :pids AND Timestamp >= :start_date ORDER BY PatientID, Timestamp DESC"),
            engine,
            params={"pids": pids, "start_date": start_date},
        )

        if wearables.empty:
            return f"No wearable data found for any of the specified patients in the last 90 days."

        wearables = _normalize_wearable_columns(wearables)
        candidate = ["Steps","HeartRate","OxygenLevel","StressLevel","SystolicBP","DiastolicBP","SleepHours"]
        features = [c for c in candidate if c in wearables.columns]
        if len(features) < 3:
            return "Not enough numeric features available to detect anomalies."

        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.1, random_state=42)
        
        results = []
        for pid, group in wearables.groupby("PatientID"):
            if len(group) < 10: # Not enough data to find anomalies
                results.append(f"--- Anomalies for patient {pid}: Not enough recent data to analyze. ---")
                continue

            X = group[features].fillna(method="ffill").fillna(method="bfill").fillna(0)
            group["anomaly"] = iso.fit_predict(X)
            anomalies = group[group["anomaly"] == -1].copy()

            if anomalies.empty:
                results.append(f"--- No anomalies detected for patient {pid} in the last 90 days. ---")
                continue
            
            anomalies["reason"] = anomalies.apply(explain_anomaly, axis=1)
            reasons = "; ".join(anomalies["reason"].tolist()[:5])
            results.append(f"--- Anomalies for patient {pid}: {len(anomalies)} unusual records found. Recent reasons: {reasons} ---")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error: {e}"

@tool("abnormal_patients", return_direct=True)
def abnormal_patients_tool(limit: str = "10") -> str:
    """
    List patients whose latest wearable reading is currently abnormal (system-wide).
    Flags: HTN (BP>140/90), LowO2 (<92), Tachy (HR>110), LowSleep (<4h).
    """
    try:
        lim = int(limit) if str(limit).strip().isdigit() else 10
        latest = pd.read_sql(
            text("""
                SELECT w.*
                FROM WearableData w
                JOIN (
                    SELECT PatientID, MAX(Timestamp) AS MaxTS
                    FROM WearableData
                    GROUP BY PatientID
                ) m ON w.PatientID = m.PatientID AND w.Timestamp = m.MaxTS
            """),
            engine,
        )
        if latest.empty:
            return "No wearable data found."

        latest = _normalize_wearable_columns(latest)
        pts = pd.read_sql(text("SELECT PatientID, Name, BMI FROM Patients"), engine)
        df = latest.merge(pts, on="PatientID", how="left")

        # Flags
        df["HTN"] = (df.get("SystolicBP", pd.Series(dtype='float64')) > 140) | (df.get("DiastolicBP", pd.Series(dtype='float64')) > 90)
        df["LowO2"] = df.get("OxygenLevel", pd.Series(dtype='float64')) < 92
        df["Tachy"] = df.get("HeartRate", pd.Series(dtype='float64')) > 110
        df["LowSleep"] = df.get("SleepHours", pd.Series(dtype='float64')) < 4

        df["flags"] = df[["HTN","LowO2","Tachy","LowSleep"]].fillna(False).apply(
            lambda r: ", ".join([k for k, v in zip(["HTN","LowO2","Tachy","LowSleep"], r) if v]), axis=1
        )
        flagged = df[df["flags"].astype(str) != ""].copy()
        if flagged.empty:
            return "No patients with abnormal latest readings."

        flagged = flagged.sort_values(["Timestamp"], ascending=False).head(lim)
        lines = [
            f"{int(r.PatientID)} | {r.get('Name','Unknown')} | Flags: {r.flags} | "
            f"BP={int(r.SystolicBP) if pd.notna(r.SystolicBP) else 'n/a'}/"
            f"{int(r.DiastolicBP) if pd.notna(r.DiastolicBP) else 'n/a'}, "
            f"HR={int(r.HeartRate) if pd.notna(r.HeartRate) else 'n/a'}, "
            f"SpO2={r.OxygenLevel if pd.notna(r.OxygenLevel) else 'n/a'}%, "
            f"Sleep={r.SleepHours if pd.notna(r.SleepHours) else 'n/a'}h, "
            f"@{r.Timestamp}"
            for _, r in flagged.iterrows()
        ]
        return f"Abnormal patients (showing up to {lim}):\n" + "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"