# server.py (Updated)
import flwr as fl
import numpy as np
from utils import get_model # We need the model architecture to save weights into it

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # 1. Run the standard aggregation (math averaging)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            print(f"Round {server_round} finished. Saving global model...")
            
            # 2. Convert "Parameters" (bytes) to "Weights" (numpy arrays)
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # 3. Load weights into Keras model
            model = get_model()
            model.set_weights(aggregated_weights)

            # 4. Save the model to disk
            model.save("global_model.keras")
            
        return aggregated_parameters, aggregated_metrics

# Initialize the CUSTOM strategy
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
)

print("Server running... Model will be saved as 'global_model.keras' after rounds.")

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)