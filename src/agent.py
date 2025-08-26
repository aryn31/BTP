# src/agent.py
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType

# Import tools from agent3 now, including the new summary tool
# <<< CHANGED from .agent2 to .agent3
from .agent3 import (
    risk_score_tool,
    detect_anomalies_tool,
    abnormal_patients_tool,
    summarize_patient_conditions_tool, # <<< ADDED the new tool
)

# Load local Mistral (make sure `ollama run mistral` works on your machine)
llm = Ollama(model="mistral", temperature=0)

# Register all available tools
tools = [
    risk_score_tool,
    detect_anomalies_tool,
    abnormal_patients_tool,
    summarize_patient_conditions_tool, # <<< ADDED the new tool to the list
]

# Build agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def run_agent(query: str):
    # agent.run is fine; agent.invoke is the newer API if you prefer
    return agent.run(query)