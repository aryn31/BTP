# agent_backend.py
from typing import Optional
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import os

# === DB Connection ===
DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
engine = create_engine(DB_URI)

# === Scheduler ===
scheduler = BackgroundScheduler()
scheduler.start()


# === Immediate Escalation (For High Risk & Critical) ===
def direct_escalate_family(alert_id: int, patient_id: int):
    """
    IMMEDIATE escalation to family (For High Risk).
    Skips the waiting period.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO AgentFamilyMessages
            (AlertID, PatientID, Sender, MessageType, Message)
            VALUES (:aid, :pid, 'Agent', 'ESCALATION', 'High Risk detected. Immediate family notification sent.')
        """), {"aid": alert_id, "pid": patient_id})
        
        print(f"🚨 IMMEDIATE Escalation to Family for Alert {alert_id}")

def direct_escalate_doctor(alert_id: int, patient_id: int, doctor_id: int):
    """
    IMMEDIATE escalation to doctor (For Critical).
    Skips the waiting period and family check.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO AgentDoctorMessages
            (AlertID, PatientID, DoctorID, Sender, MessageType, Message)
            VALUES (:aid, :pid, :did, 'Agent', 'ESCALATION', 'CRITICAL CONDITION DETECTED. Immediate doctor review required.')
        """), {"aid": alert_id, "pid": patient_id, "did": doctor_id})
        
        print(f"🚑 IMMEDIATE Escalation to Doctor {doctor_id} for Alert {alert_id}")

# === Alert creation ===
def create_alert(patient_id: int, message: str) -> int:
    """
    Create a new alert for a patient and log initial agent message.
    """
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO Alerts (PatientID) VALUES (:pid)"), {"pid": patient_id})
        alert_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()

        # Log agent's initial alert to patient
        conn.execute(text("""
            INSERT INTO AgentPatientMessages (AlertID, PatientID, Sender, MessageType, Message)
            VALUES (:aid, :pid, 'Agent', 'ALERT', :msg)
        """), {"aid": alert_id, "pid": patient_id, "msg": message})

    return alert_id

# === Escalation functions ===
def escalate_to_family(alert_id: int, patient_id: int, doctor_id: int):
    """
    Escalate an unresolved alert to the family member.
    """
    with engine.begin() as conn:
        # Check if patient responded
        patient_resp = conn.execute(text("""
            SELECT COUNT(*) FROM AgentPatientMessages
            WHERE AlertID=:aid AND MessageType='RESPONSE'
        """), {"aid": alert_id}).scalar()

        if patient_resp == 0:
            # Log to family member
            conn.execute(text("""
                INSERT INTO AgentFamilyMessages
                (AlertID, PatientID, Sender, MessageType, Message)
                VALUES (:aid, :pid, 'Agent', 'ESCALATION', 'Patient did not respond. Escalating to family member.')
            """), {"aid": alert_id, "pid": patient_id})

            # Schedule doctor escalation after 3 minutes
            scheduler.add_job(
                escalate_to_doctor,
                'date',
                run_date=datetime.now() + timedelta(minutes=3),
                args=[alert_id, patient_id, doctor_id]
            )

            print(f"🚨 Escalated Alert {alert_id} to Family Member")

def escalate_to_doctor(alert_id: int, patient_id: int, doctor_id: int):
    """
    Escalate an unresolved alert to the doctor if neither patient nor family responded.
    """
    with engine.begin() as conn:
        # Check responses from patient & family
        total_resp = conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT * FROM AgentPatientMessages WHERE AlertID=:aid AND MessageType='RESPONSE'
                UNION ALL
                SELECT * FROM AgentFamilyMessages WHERE AlertID=:aid AND MessageType='RESPONSE'
            ) AS combined
        """), {"aid": alert_id}).scalar()

        if total_resp == 0:
            # Log escalation to doctor
            conn.execute(text("""
                INSERT INTO AgentDoctorMessages
                (AlertID, PatientID, DoctorID, Sender, MessageType, Message)
                VALUES (:aid, :pid, :did, 'Agent', 'ESCALATION', 'No response from patient/family. Please review.')
            """), {"aid": alert_id, "pid": patient_id, "did": doctor_id})

            print(f"🚨 Escalated Alert {alert_id} to Doctor {doctor_id}")

# === Schedule escalation ===
def schedule_escalation(alert_id: int, patient_id: int, doctor_id: int, 
                        timeout_patient: int = 3, timeout_family: int = 3):
    scheduler.add_job(
        escalate_to_family,
        'date',
        run_date=datetime.now() + timedelta(minutes=timeout_patient),
        args=[alert_id, patient_id, doctor_id]
    )
    print(f"⏳ Escalation scheduled for Alert {alert_id}: Patient → Family → Doctor")
