# import os
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import flwr as fl
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # ==========================================
# # 1. SIMULATED DATA LOADING (The "Virtual World")
# # ==========================================
# def load_partition(client_id):
#     """
#     This function simulates being a specific client.
#     It loads ONLY that client's slice of data.
#     """
#     # Load the FULL dataset (In real life, you wouldn't have this)
#     # Make sure 'patient_summary.csv' is in the same folder
#     df = pd.read_csv('patient_summary.csv') 
    
#     # --- PREPROCESSING (Must match your previous logic) ---
#     df = df.drop(columns=['PatientID', 'DOB'], errors='ignore')
#     df = df.dropna()
    
#     # Encode Gender & Target
#     df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
#     target_map = {'Fit': 0, 'Slightly Unfit': 1, 'Severe': 2}
#     df['FitnessCategory'] = df['FitnessCategory'].map(target_map)
    
#     # Split Features/Target
#     X = df.drop('FitnessCategory', axis=1).values
#     y = df['FitnessCategory'].values
    
#     # --- THE PARTITIONING TRICK ---
#     # We split the data into 3 equal chunks to simulate 3 clients
#     total_samples = len(df)
#     partition_size = total_samples // 3
    
#     start_idx = int(client_id) * partition_size
#     end_idx = start_idx + partition_size
    
#     # Return only THIS client's slice
#     X_part = X[start_idx:end_idx]
#     y_part = y[start_idx:end_idx]
    
#     # Scale (Fitting scaler on local data only to be realistic)
#     scaler = StandardScaler()
#     X_part = scaler.fit_transform(X_part)
    
#     return X_part, y_part

# # ==========================================
# # 2. MODEL DEFINITION
# # ==========================================
# def get_model(input_shape=8):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(input_shape,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(3, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# # ==========================================
# # 3. THE FLOWER CLIENT
# # ==========================================
# class SimulatedClient(fl.client.NumPyClient):
#     def __init__(self, client_id):
#         self.client_id = client_id
#         self.model = get_model()
#         self.x_train, self.y_train = load_partition(client_id)

#     def get_parameters(self, config):
#         return self.model.get_weights()

#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)
#         # Train on the simulated local partition
#         self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=16, verbose=0)
#         return self.model.get_weights(), len(self.x_train), {}

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         loss, acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
#         return loss, len(self.x_train), {"accuracy": acc}

# # ==========================================
# # 4. THE SIMULATION ENGINE
# # ==========================================
# def client_fn(cid: str):
#     """Flower calls this function to spawn a client."""
#     return SimulatedClient(client_id=cid)

# if __name__ == "__main__":
#     # We simulate 3 clients
#     # The server will sample 100% of them (fraction_fit=1.0)
#     print("Starting Simulation with 3 Virtual Clients...")
    
#     fl.simulation.start_simulation(
#         client_fn=client_fn,
#         num_clients=3,
#         config=fl.server.ServerConfig(num_rounds=5),
#         strategy=fl.server.strategy.FedAvg(
#             fraction_fit=1.0, 
#             min_fit_clients=3
#         )
#     )

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import flwr as fl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

# ==========================================
# 1. SIMULATED DATA LOADING (The "Virtual World")
# ==========================================
def load_partition(client_id):
    """
    Loads data for a simulated client and converts DOB → Age.
    """
    # Load full dataset
    df = pd.read_csv('patient_summary.csv') 
    
    # ------------------------------------------
    # Convert DOB → Age
    # ------------------------------------------
    if "DOB" in df.columns:
        # Convert to datetime
        df["DOB"] = pd.to_datetime(df["DOB"], errors='coerce')

        # Compute Age (in years)
        today = pd.Timestamp.today()
        df["Age"] = (today - df["DOB"]).dt.days // 365

    # Drop original DOB column (not needed after conversion)
    df = df.drop(columns=['PatientID', 'DOB'], errors='ignore')

    # Drop missing
    df = df.dropna()
    
    # Encode Gender & Target
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    target_map = {'Fit': 0, 'Slightly Unfit': 1, 'Severe': 2}
    df['FitnessCategory'] = df['FitnessCategory'].map(target_map)
    
    # Split X and y
    X = df.drop('FitnessCategory', axis=1).values
    y = df['FitnessCategory'].values
    
    # ------------------------------------------
    # Partition data into 3 clients
    # ------------------------------------------
    total_samples = len(df)
    partition_size = total_samples // 3
    
    start_idx = int(client_id) * partition_size
    end_idx = start_idx + partition_size

    X_part = X[start_idx:end_idx]
    y_part = y[start_idx:end_idx]
    
    # Scale locally (each client scales its own data)
    scaler = StandardScaler()
    X_part = scaler.fit_transform(X_part)
    
    return X_part, y_part

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
def get_model(input_shape=8):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# 3. THE FLOWER CLIENT
# ==========================================
class SimulatedClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.x_train, self.y_train = load_partition(client_id)
        input_shape = self.x_train.shape[1]
        self.model = get_model(input_shape=input_shape)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=16, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"accuracy": acc}

# ==========================================
# 4. THE SIMULATION ENGINE
# ==========================================
def client_fn(cid: str):
    return SimulatedClient(client_id=cid)

if __name__ == "__main__":
    print("Starting Simulation with DOB → Age conversion enabled...")
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0, 
            min_fit_clients=3
        )
    )
