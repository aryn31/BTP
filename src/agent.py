# src/agent.py
# <<< FIXED DEPRECATION WARNING >>>
# The new recommended way to import and use Ollama
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType

from .agent4 import (
    get_patient_medications_tool,
    get_lab_results_tool,
    get_encounter_notes_tool,
    summarize_patient_conditions_tool,
    risk_score_tool,
    detect_anomalies_tool,
    abnormal_patients_tool,
)

from .email_workflow_tool import escalate_via_email

# <<< FIXED DEPRECATION WARNING >>>
# Initialize with the new class name
llm = OllamaLLM(model="mistral", temperature=0)

tools = [
    get_patient_medications_tool,
    get_lab_results_tool,
    get_encounter_notes_tool,
    summarize_patient_conditions_tool,
    risk_score_tool,
    detect_anomalies_tool,
    abnormal_patients_tool,
    escalate_via_email,
]

# The LangGraph warning is a general recommendation for new projects.
# Your current agent initialization is still functional and correct for this setup.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def run_agent(query: str):
    return agent.run(query)