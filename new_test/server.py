import flwr as fl
import torch
import os
import numpy as np
from collections import OrderedDict
from typing import List, Tuple
from model import HealthClassifier

# --- 1. Metric Aggregation Function ---
def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """
    Averages the 'accuracy' sent by clients, weighted by the number of examples they have.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    if not examples: return {"accuracy": 0}
    
    # Return global accuracy
    return {"accuracy": sum(accuracies) / sum(examples)}

# --- 2. Custom Strategy to Save Model ---
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            print(f"💾 Saving updated global model (Round {server_round})...")
            # Convert Flower Params -> NumPy -> PyTorch
            params = fl.common.parameters_to_ndarrays(aggregated_parameters)
            net = HealthClassifier()
            params_dict = zip(net.state_dict().keys(), params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            torch.save(state_dict, "federated_model.pth")
            
        return aggregated_parameters, aggregated_metrics

def load_initial_parameters():
    """Check for existing model to resume training."""
    if os.path.exists("federated_model.pth"):
        print("🔄 Found 'federated_model.pth'. Resuming training!")
        net = HealthClassifier()
        net.load_state_dict(torch.load("federated_model.pth"))
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return fl.common.ndarrays_to_parameters(weights)
    else:
        print("🆕 No saved model found. Starting fresh.")
        return None

if __name__ == "__main__":
    initial_params = load_initial_parameters()
    
    strategy = SaveModelStrategy(
        min_fit_clients=1,       # <--- Change to 1
        min_available_clients=1, # <--- Change to 1
        min_evaluate_clients=1,
        fraction_fit=1.0,
        initial_parameters=initial_params,
        
        # LINK THE METRIC FUNCTION HERE:
        evaluate_metrics_aggregation_fn=weighted_average 
    )

    print("🚀 Starting Federated Server on Port 8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=10), 
        strategy=strategy
    )