# src/agent.py
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from sqlalchemy import create_engine, text
import os, re
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# Import new backend functions
from .agent_backend import create_alert, schedule_escalation

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

# === Scheduler for internal use (not strictly needed as agent_backend uses its own) ===
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

# === Updated run_agent ===
def run_agent(query: str):
    """
    Process a user query via the LLM, extract patient info, 
    check risk scores, and create alerts with three-stage escalation.
    """
    response = agent.run(query)

    # Extract patient ID from query
    match = re.search(r"patient\s+(\d+)", query, re.IGNORECASE)
    patient_id = int(match.group(1)) if match else None

    if patient_id:
        # Parse risk_score output if present in LLM response
        if "Risk scores" in response:
            risk_match = re.search(r"\{.*\}", response, re.DOTALL)
            if risk_match:
                try:
                    risks = eval(risk_match.group())  # safe controlled format
                    for condition, score in risks.items():
                        if score >= 0.7:  # threshold for alert
                            alert_msg = f"High {condition} risk detected (score={score}). Please confirm."
                            alert_id = create_alert(patient_id, alert_msg)

                            # For now, assume DoctorID=1 (can later fetch from DoctorAllocation table)
                            schedule_escalation(alert_id, patient_id, doctor_id=1)

                            print(f"⚡ Alert triggered for patient {patient_id}, AlertID={alert_id}")
                except Exception as e:
                    print("⚠️ Could not parse risks:", e)

    return response
