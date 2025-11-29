# server.py (Run ONLY on Machine A)
import flwr as fl
from utils import get_model

# Define Strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,         # Use 100% of connected clients
    min_fit_clients=1,        # Allow training with just 1 client for testing
    min_evaluate_clients=1,
    min_available_clients=1,  # Start as soon as 1 client connects
)

print("Server is starting...")
print("Listening on Port 8080. Tell your clients to connect to my IP!")

fl.server.start_server(
    server_address="0.0.0.0:8080", # Listen on all interfaces
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)