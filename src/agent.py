# src/agent.py
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from sqlalchemy import create_engine, text
import os, re
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from .agent4 import (
    get_patient_medications_tool,
    get_lab_results_tool,
    get_encounter_notes_tool,
    summarize_patient_conditions_tool,
    risk_score_tool,
    detect_anomalies_tool,
    abnormal_patients_tool,
)

# === DB Connection ===
DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
engine = create_engine(DB_URI)

# === Scheduler for escalation ===
scheduler = BackgroundScheduler()
scheduler.start()

# === LLM + Tools ===
llm = OllamaLLM(model="mistral", temperature=0)
tools = [
    get_patient_medications_tool,
    get_lab_results_tool,
    get_encounter_notes_tool,
    summarize_patient_conditions_tool,
    risk_score_tool,
    detect_anomalies_tool,
    abnormal_patients_tool,
]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# === Helper DB functions ===
def create_alert(patient_id: int, message: str):
    with engine.begin() as conn:
        alert_id = conn.execute(
            text("INSERT INTO Alerts (PatientID) VALUES (:pid)"),
            {"pid": patient_id}
        ).lastrowid
        conn.execute(
            text("""
                INSERT INTO AgentPatientMessages
                (AlertID, PatientID, Sender, MessageType, Message)
                VALUES (:aid, :pid, 'Agent', 'ALERT', :msg)
            """),
            {"aid": alert_id, "pid": patient_id, "msg": message}
        )
    return alert_id


def log_escalation(patient_id: int, doctor_id: int, alert_id: int):
    with engine.begin() as conn:
        # Log in Agent ↔ Patient
        conn.execute(
            text("""
                INSERT INTO AgentPatientMessages
                (AlertID, PatientID, Sender, MessageType, Message)
                VALUES (:aid, :pid, 'Agent', 'ESCALATION',
                        'Patient did not respond. Escalating to doctor.')
            """),
            {"aid": alert_id, "pid": patient_id}
        )
        # Log in Agent ↔ Doctor
        conn.execute(
            text("""
                INSERT INTO AgentDoctorMessages
                (AlertID, PatientID, DoctorID, Sender, MessageType, Message)
                VALUES (:aid, :pid, :did, 'Agent', 'ESCALATION',
                        'Patient did not respond. Please follow up.')
            """),
            {"aid": alert_id, "pid": patient_id, "did": doctor_id}
        )
    print(f"🚨 Escalated Alert {alert_id} for Patient {patient_id} to Doctor {doctor_id}")


# === Escalation check job ===
def check_and_escalate(alert_id: int, patient_id: int, doctor_id: int):
    with engine.begin() as conn:
        resp = conn.execute(
            text("""
                SELECT COUNT(*) FROM AgentPatientMessages
                WHERE AlertID=:aid AND MessageType='RESPONSE'
            """),
            {"aid": alert_id}
        ).scalar()
    if resp == 0:  # no patient response
        log_escalation(patient_id, doctor_id, alert_id)
    


def schedule_escalation(alert_id: int, patient_id: int, doctor_id: int, timeout=5):
    run_at = datetime.now() + timedelta(minutes=timeout)
    scheduler.add_job(
        check_and_escalate,
        "date",
        run_date=run_at,
        args=[alert_id, patient_id, doctor_id]
    )
    print(f"⏳ Escalation scheduled for Alert {alert_id} at {run_at}")


# === Updated run_agent ===
def run_agent(query: str):
    response = agent.run(query)

    # Extract patient ID from query
    match = re.search(r"patient\s+(\d+)", query, re.IGNORECASE)
    patient_id = int(match.group(1)) if match else None

    # Parse risk_score output if present
    if patient_id and "Risk scores" in response:
        risk_match = re.search(r"\{.*\}", response, re.DOTALL)
        if risk_match:
            try:
                risks = eval(risk_match.group())  # safe controlled format
                for condition, score in risks.items():
                    if score >= 0.7:  # threshold
                        alert_msg = f"High {condition} risk detected (score={score}). Please confirm."
                        alert_id = create_alert(patient_id, alert_msg)

                        # For now, assume DoctorID=1 (later can map via DoctorAllocation table)
                        schedule_escalation(alert_id, patient_id, doctor_id=1, timeout=2)
                        print(f"⚡ Alert triggered for patient {patient_id}, AlertID={alert_id}")
            except Exception as e:
                print("⚠️ Could not parse risks:", e)

    return response
