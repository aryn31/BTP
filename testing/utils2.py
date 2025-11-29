# utils.py (Save on ALL machines)
import pandas as pd
import numpy as np
import tensorflow as tf

def get_model(input_shape):
    """
    Upgraded Model: Wider, better activation, and AdamW optimizer.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # Layer 1: Wider (256) + GELU (Modern activation)
        tf.keras.layers.Dense(256, activation='gelu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Layer 2: 128 Neurons
        tf.keras.layers.Dense(128, activation='gelu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Layer 3: 64 Neurons (Added depth)
        tf.keras.layers.Dense(64, activation='gelu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # Use AdamW (Weight Decay) for better generalization
    opt = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
    
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def load_local_data(csv_path):
    """
    Loads data with GLOBAL scaling and Feature Engineering.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {csv_path}")
        return None, None

    # 1. Clean
    df = df.drop(columns=['PatientID', 'DOB'], errors='ignore')
    df = df.dropna()

    # 2. Hardcoded Mappings
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    target_map = {'Fit': 0, 'Slightly Unfit': 1, 'Severe': 2}
    df['FitnessCategory'] = df['FitnessCategory'].map(target_map)

    # 3. FEATURE ENGINEERING (New Inputs)
    # Give the model explicit medical hints
    df['PulsePressure'] = df['BP_Sys'] - df['BP_Dia']
    df['MAP'] = df['BP_Dia'] + (df['PulsePressure'] / 3) # Mean Arterial Pressure

    # 4. Separate Target
    y = df['FitnessCategory'].values
    X_raw = df.drop('FitnessCategory', axis=1)

    # 5. FIXED GLOBAL SCALING (Crucial for Federated Learning)
    # Instead of fitting a scaler locally, we divide by reasonable max values.
    # This ensures "Age 50" looks the same on Client A and Client B.
    X = X_raw.copy()
    
    # Define approximate max values for medical data
    X['Age'] = X['Age'] / 100.0
    X['HeartRate_bpm'] = X['HeartRate_bpm'] / 200.0
    X['SpO2'] = X['SpO2'] / 100.0
    X['StressLevel'] = X['StressLevel'] / 10.0 # Assuming scale 1-10
    X['BP_Sys'] = X['BP_Sys'] / 200.0
    X['BP_Dia'] = X['BP_Dia'] / 150.0
    X['BMI'] = X['BMI'] / 50.0
    X['PulsePressure'] = X['PulsePressure'] / 100.0
    X['MAP'] = X['MAP'] / 150.0
    # Gender is already 0/1, so no scaling needed

    return X.values.astype('float32'), y