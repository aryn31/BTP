# server.py (Updated with Saving)
import flwr as fl
import numpy as np
from utils2 import get_model

# 1. Define a strategy that saves the model
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Run the standard math (Averaging)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            print(f"Round {server_round} complete. Saving weights...")
            
            # Convert bytes -> numpy arrays
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Load into model structure
            model = get_model(input_shape=10)
            # model = get_model()
            model.set_weights(weights)
            
            # SAVE TO DISK
            model.save(f"/Users/aryansheel/Desktop/untitled_folder/saved_models/federated_model_round_{server_round}.keras")
            print(f"Saved: federated_model_round_{server_round}.keras")
            
        return aggregated_parameters, aggregated_metrics

# 2. Use this custom strategy
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1, # wait for both clients to train
    min_evaluate_clients=1, # wait for both clients to test
    min_available_clients=1, # don't start until these much are online
)

print("Server running... Models will be saved to this folder.")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)