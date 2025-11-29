# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def get_model(input_shape=8):
    """
    Defines a simple Neural Network for tabular data.
    Input Shape 8: Age, Gender, HR, SpO2, Stress, BP_Sys, BP_Dia, BMI
    Output Shape 3: Fit, Slightly Unfit, Severe
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') # 3 Classes
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def load_data(csv_path):
    """
    Loads data and forces consistent encoding across all clients.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Drop ID columns
    df = df.drop(columns=['PatientID', 'DOB'], errors='ignore')
    
    # 2. Hardcoded Target Mapping (CRITICAL for FL)
    # If we don't hardcode this, Client A might think 0=Fit 
    # and Client B might think 0=Severe.
    target_map = {'Fit': 0, 'Slightly Unfit': 1, 'Severe': 2}
    df['FitnessCategory'] = df['FitnessCategory'].map(target_map)
    
    # 3. Hardcoded Gender Mapping
    gender_map = {'Male': 0, 'Female': 1}
    df['Gender'] = df['Gender'].map(gender_map)
    
    # 4. Drop missing values
    df = df.dropna()
    
    # 5. Split features and target
    X = df.drop('FitnessCategory', axis=1)
    y = df['FitnessCategory']
    
    # 6. Scale Features (Simple standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values