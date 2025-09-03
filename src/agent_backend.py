from typing import Optional
from langchain.tools import tool
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import time
import os

# === DB Connection ===
DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
engine = create_engine(DB_URI)

# Create a new alert
@tool
def create_alert(patient_id: int, message: str) -> int:
    """Create a new alert for a patient and log initial agent message."""
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO Alerts (PatientID) VALUES (:pid)"), {"pid": patient_id})
        alert_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()

        conn.execute(text("""
            INSERT INTO AgentPatientMessages (AlertID, PatientID, Sender, MessageType, Message)
            VALUES (:aid, :pid, 'Agent', 'ALERT', :msg)
        """), {"aid": alert_id, "pid": patient_id, "msg": message})
    return alert_id

# Check if patient responded within N minutes
@tool
def check_patient_response(alert_id: int, timeout_minutes: int = 5) -> bool:
    """Wait for a patient response within the timeout. Return True if responded, False otherwise."""
    deadline = datetime.now() + timedelta(minutes=timeout_minutes)
    while datetime.now() < deadline:
        with engine.begin() as conn:
            res = conn.execute(text("""
                SELECT COUNT(*) FROM AgentPatientMessages
                WHERE AlertID=:aid AND Sender='Patient' AND MessageType='RESPONSE'
            """), {"aid": alert_id}).scalar()
        if res > 0:
            return True
        time.sleep(10)
    return False

# Escalate to doctor
@tool
def escalate_to_doctor(alert_id: int, patient_id: int, doctor_id: int):
    """Escalate an unresolved alert to a doctor."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO AgentPatientMessages (AlertID, PatientID, Sender, MessageType, Message)
            VALUES (:aid, :pid, 'Agent', 'ESCALATION', 'Patient did not respond, escalating to doctor.')
        """), {"aid": alert_id, "pid": patient_id})

        conn.execute(text("""
            INSERT INTO AgentDoctorMessages (AlertID, PatientID, DoctorID, Sender, MessageType, Message)
            VALUES (:aid, :pid, :did, 'Agent', 'ESCALATION', 'Patient did not respond, please review.')
        """), {"aid": alert_id, "pid": patient_id, "did": doctor_id})

    return f"Escalated alert {alert_id} to doctor {doctor_id}"
