# src/agent.py
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from sqlalchemy import create_engine, text
import os, re
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# Import new backend functions
from .agent_backend import create_alert, schedule_escalation, direct_escalate_family, direct_escalate_doctor

from .agent4 import (
    get_patient_medications_tool,
    get_lab_results_tool,
    get_encounter_notes_tool,
    assess_health_category_tool,
    summarize_patient_conditions_tool,
    risk_score_tool,
    detect_anomalies_tool,
    abnormal_patients_tool,
)

DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
engine = create_engine(DB_URI)

scheduler = BackgroundScheduler()
scheduler.start()

llm = OllamaLLM(model="mistral", temperature=0)
tools = [
    assess_health_category_tool,
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

# === Updated run_agent ===
# def run_agent(query: str):
#     """
#     Process a user query via the LLM, extract patient info, 
#     check risk scores, and create alerts with three-stage escalation.
#     """
#     response = agent.run(query)

#     match = re.search(r"patient\s+(\d+)", query, re.IGNORECASE)
#     patient_id = int(match.group(1)) if match else None

#     if patient_id:

#         if "Risk scores" in response:
#             risk_match = re.search(r"\{.*\}", response, re.DOTALL)
#             if risk_match:
#                 try:
#                     risks = eval(risk_match.group())  
#                     for condition, score in risks.items():
#                         if score >= 0.7:  
#                             alert_msg = f"High {condition} risk detected (score={score}). Please confirm."
#                             alert_id = create_alert(patient_id, alert_msg)

#                             schedule_escalation(alert_id, patient_id, doctor_id=1)

#                             print(f"⚡ Alert triggered for patient {patient_id}, AlertID={alert_id}")
#                 except Exception as e:
#                     print("⚠️ Could not parse risks:", e)

#     return response

def run_agent(query: str):
    """
    1. Extract PatientID.
    2. Run Federated Model (Assessment Tool).
    3. Execute Logic based on Category (Optimal, Elevated, High, Critical).
    """
    
    # 1. Extract Patient ID using Regex (Fastest way)
    match = re.search(r"patient\s+(\d+)", query, re.IGNORECASE)
    patient_id = int(match.group(1)) if match else None

    # Run the generic LLM response first (to chat with user)
    response = agent.run(query)

    if patient_id:
        print(f"🔎 Analyzing Patient {patient_id}...")
        
        # 2. Call the tool DIRECTLY (Bypassing LLM for the critical logic)
        # This ensures the logic is deterministic and not subject to LLM hallucinations
        category = assess_health_category_tool.func(str(patient_id))
        
        print(f"🤖 Federated Model Prediction: {category}")

        # 3. Apply the Project Logic
        if category == "Optimal":
            print(f"✅ Status Optimal. No action taken.")
            
        elif category == "Elevated Risk":
            # Logic: Alert Patient -> Wait -> Family -> Wait -> Doctor
            msg = f"Health Alert: Elevated risk detected. Please take medication/rest."
            alert_id = create_alert(patient_id, msg)
            schedule_escalation(alert_id, patient_id, doctor_id=1)
            print(f"⚠️ Elevated Risk: Alert {alert_id} sent to Patient.")

        elif category == "High Risk":
            # Logic: Alert Patient AND Alert Family IMMEDIATELY
            msg = f"URGENT: High Health Risk detected."
            alert_id = create_alert(patient_id, msg)
            direct_escalate_family(alert_id, patient_id) # <--- Immediate
            # Still schedule doctor backup if family doesn't respond
            schedule_escalation(alert_id, patient_id, doctor_id=1) 
            print(f"🚨 High Risk: Escalated to Family immediately.")

        elif category == "Critical":
            # Logic: Alert Doctor IMMEDIATELY
            msg = f"CRITICAL: Emergency assistance required."
            alert_id = create_alert(patient_id, msg)
            direct_escalate_doctor(alert_id, patient_id, doctor_id=1) # <--- Immediate
            print(f"🚑 CRITICAL: Escalated to Doctor immediately.")

    return response