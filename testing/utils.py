# utils.py (Save on ALL machines)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def get_model(input_shape=8):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # Layer 1: Wider + Batch Norm + Dropout
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(), # <--- NEW: Stabilizes training
        tf.keras.layers.Dropout(0.3),         # <--- NEW: Prevents overfitting
        
        # Layer 2: Deepening the network
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Layer 3: Output
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # Lower the learning rate slightly for stability (0.001 -> 0.0005)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def load_local_data(csv_path):
    """
    Loads the specific CSV file located on this machine.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {csv_path}. Make sure the file exists.")
        return None, None

    # --- CONSTANT PREPROCESSING ---
    # 1. Drop unused
    df = df.drop(columns=['PatientID', 'DOB'], errors='ignore')
    df = df.dropna()

    # 2. Hardcoded Encodings (Must be consistent across network)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    target_map = {'Fit': 0, 'Slightly Unfit': 1, 'Severe': 2}
    df['FitnessCategory'] = df['FitnessCategory'].map(target_map)

    # 3. Split
    X = df.drop('FitnessCategory', axis=1)
    y = df['FitnessCategory']

    # 4. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values