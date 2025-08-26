# main.py
from src.agent import run_agent

def main():
    print("🩺 Medical Risk & Anomaly Detection Agent (powered by Mistral via Ollama)\n")
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye! 👋")
            break
        try:
            result = run_agent(query)
            print("\n✅ Agent Response:\n", result, "\n")
        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    main()
