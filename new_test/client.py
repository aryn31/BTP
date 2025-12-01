import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
from model import HealthClassifier, train, test # Import shared code

# 1. Load Data
if len(sys.argv) < 2:
    print("Usage: python client.py <dataset_filename>")
    sys.exit(1)

dataset_path = sys.argv[1]
print(f"🏥 Loading local data from {dataset_path}...")
df = pd.read_csv(dataset_path)

# 2. Preprocessing
features = ["Age", "BMI", "HeartRate_bpm", "SpO2", "BP_Sys", "BP_Dia", "StressLevel"]
target = "RiskCategory"

# Encode Targets (alphabetical order: Critical=0, Elevated=1, High=2, Optimal=3)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df[target])

# Scale Features (Important for Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Initialize Model
net = HealthClassifier()

# 4. Define Flower Client
class HospitalClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Send local weights to server
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        # Update local model with Global weights
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train on local data
        train(net, train_loader, epochs=5)
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, train_loader)
        return float(loss), len(train_loader.dataset), {"accuracy": float(accuracy)}

# 5. Start Client
print("📡 Connecting to Federated Server...")
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=HospitalClient())