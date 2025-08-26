# src/agent.py
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType

# Import tools from agent1
# from .agent1 import risk_score_tool, detect_anomalies_tool, abnormal_patients_tool
from .agent2 import risk_score_tool, detect_anomalies_tool, abnormal_patients_tool

# Load local Mistral (make sure `ollama run mistral` works on your machine)
llm = Ollama(model="mistral", temperature=0)

# Register tools
tools = [risk_score_tool, detect_anomalies_tool, abnormal_patients_tool]

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
