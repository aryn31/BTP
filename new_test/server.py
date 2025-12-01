# server.py
import flwr as fl
import torch
import os
import numpy as np
from collections import OrderedDict
from model import HealthClassifier

# Standard strategy to save the model after rounds
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            print(f"💾 Saving updated global model (Round {server_round})...")
            params = fl.common.parameters_to_ndarrays(aggregated_parameters)
            net = HealthClassifier()
            params_dict = zip(net.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            torch.save(state_dict, "federated_model.pth")
        return aggregated_parameters, aggregated_metrics

def load_initial_parameters():
    """
    Checks if a saved model exists.
    If yes, converts it to Flower parameters for the server to use as a starting point.
    """
    if os.path.exists("federated_model.pth"):
        print("🔄 Found existing 'federated_model.pth'. Resuming training from saved state!")
        
        # 1. Load the PyTorch model
        net = HealthClassifier()
        net.load_state_dict(torch.load("federated_model.pth"))
        
        # 2. Extract weights as NumPy arrays (what Flower expects)
        # We must use .cpu().numpy() to ensure they are standard arrays
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        
        # 3. Convert to Flower Parameters object
        return fl.common.ndarrays_to_parameters(weights)
    else:
        print("🆕 No saved model found. Starting fresh (Random Initialization).")
        return None

if __name__ == "__main__":
    
    # 1. Load the saved model (if it exists)
    initial_params = load_initial_parameters()

    # 2. Define Strategy
    strategy = SaveModelStrategy(
        min_fit_clients=2,
        min_available_clients=2,
        fraction_fit=1.0,
        initial_parameters=initial_params  # <--- PASS THE SAVED MODEL HERE
    )

    # 3. Start Server
    print("🚀 Starting Federated Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=3), 
        strategy=strategy
    )