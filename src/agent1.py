# src/agent1.py
from typing import Dict, List
from langchain.tools import tool
from sqlalchemy import create_engine, text
import pandas as pd

# === DB Connection ===
DB_URI = "mysql+pymysql://root:sql_my1country@localhost:3306/BTP"
engine = create_engine(DB_URI)

# === Utilities ===
def _clean_patient_id(PatientID) -> int:
    """Extract numeric patient ID safely from inputs like '5' or 'PatientID=5'."""
    if isinstance(PatientID, int):
        return PatientID
    if isinstance(PatientID, str):
        PatientID = PatientID.replace("PatientID=", "").strip()
        PatientID = PatientID.strip("'\"")
    return int(PatientID)

def _normalize_wearable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map various schema names to a consistent internal schema.
    Works with your original table (Sleep_hrs, HeartRate_bpm, SpO2, BP_Sys, BP_Dia)
    and with cleaner names if you later switch.
    """
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

    # Force numeric where applicable
    for col in ["Steps", "HeartRate", "OxygenLevel", "StressLevel", "SystolicBP", "DiastolicBP", "SleepHours"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Timestamp to datetime if present
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return df

# === Risk Scoring ===
def calculate_risk_scores(patient_row: pd.Series, wearable_row: pd.Series):
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
@tool("risk_score", return_direct=True)
def risk_score_tool(PatientID: str) -> str:
    """Calculate patient risk score from the *latest* wearable record and explain reasoning (MySQL)."""
    try:
        pid = _clean_patient_id(PatientID)

        # Patient row
        patient = pd.read_sql(text("SELECT * FROM Patients WHERE PatientID = :pid"), engine, params={"pid": pid})
        if patient.empty:
            return f"No patient found with ID {pid}."
        p = patient.iloc[0]

        # Latest wearable row for this patient
        wearable = pd.read_sql(
            text("""
                SELECT w.* FROM WearableData w
                JOIN (
                    SELECT PatientID, MAX(Timestamp) AS MaxTS
                    FROM WearableData
                    WHERE PatientID = :pid
                    GROUP BY PatientID
                ) m
                ON w.PatientID = m.PatientID AND w.Timestamp = m.MaxTS
            """),
            engine,
            params={"pid": pid},
        )
        if wearable.empty:
            return f"No wearable data found for patient {pid}."

        w = _normalize_wearable_columns(wearable).iloc[0]
        risks, explanations = calculate_risk_scores(p, w)

        name = p.get("Name", f"Patient {pid}")
        return (
            f"Risk scores for {name} (ID {pid}) based on latest wearable ({w.get('Timestamp', 'n/a')}):\n"
            f"{risks}\nReasons: " + "; ".join(explanations)
        )
    except Exception as e:
        return f"Error: {e}"

@tool("detect_anomalies", return_direct=True)
def detect_anomalies_tool(PatientID: str) -> str:
    """Detect anomalies in a patient's wearable time series using IsolationForest (MySQL)."""
    try:
        pid = _clean_patient_id(PatientID)

        wearables = pd.read_sql(
            text("SELECT * FROM WearableData WHERE PatientID = :pid ORDER BY Timestamp DESC"),
            engine,
            params={"pid": pid},
        )
        if wearables.empty:
            return f"No wearable data found for patient {pid}."

        wearables = _normalize_wearable_columns(wearables)
        # Choose available features
        candidate = ["Steps","HeartRate","OxygenLevel","StressLevel","SystolicBP","DiastolicBP","SleepHours"]
        features = [c for c in candidate if c in wearables.columns]
        if len(features) < 3:
            return "Not enough numeric features available to detect anomalies."

        X = wearables[features].fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Lazy import to avoid startup cost
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.1, random_state=42)
        wearables["anomaly"] = iso.fit_predict(X)

        anomalies = wearables[wearables["anomaly"] == -1].copy()
        if anomalies.empty:
            return f"No anomalies detected for patient {pid}."

        anomalies["reason"] = anomalies.apply(explain_anomaly, axis=1)
        reasons = "; ".join(anomalies["reason"].tolist()[:10])  # cap size
        return f"Anomalies for patient {pid}: {len(anomalies)} unusual records. Reasons: {reasons}"
    except Exception as e:
        return f"Error: {e}"

@tool("abnormal_patients", return_direct=True)
def abnormal_patients_tool(limit: str = "50") -> str:
    """
    List patients whose latest wearable reading looks abnormal.
    Flags: HTN (BP>140/90), LowO2 (<92), Tachy (HR>110), LowSleep (<4h).
    """
    try:
        lim = int(limit) if str(limit).strip().isdigit() else 50

        # Latest wearable per patient
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

        # Bring in names
        pts = pd.read_sql(text("SELECT PatientID, Name, BMI FROM Patients"), engine)
        df = latest.merge(pts, on="PatientID", how="left")

        # Flags
        df["HTN"] = (df.get("SystolicBP", pd.Series()) > 140) | (df.get("DiastolicBP", pd.Series()) > 90)
        df["LowO2"] = df.get("OxygenLevel", pd.Series()) < 92
        df["Tachy"] = df.get("HeartRate", pd.Series()) > 110
        df["LowSleep"] = df.get("SleepHours", pd.Series()) < 4

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
