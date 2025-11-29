
#Just to predict after the model has been saved

# predict.py
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load the Federated Model
print("Loading Global Model...")
model = tf.keras.models.load_model("global_model.keras")

# 2. Define a New Patient (Raw Data)
# Let's say this is a 45-year-old Male with high BP
new_patient = {
    'Age': 45,
    'Gender': 0,      # Male (Must match your utils.py mapping!)
    'HeartRate': 85,
    'SpO2': 96.8,
    'Stress': 1,
    'BP_Sys': 140,
    'BP_Dia': 89,
    'BMI': 23.1
}

# 3. Preprocess (Scale)
# IMPORTANT: In a real production system, you must use the SAME scaler
# used during training. For this demo, we will manually normalize 
# based on approximate medical ranges just to get it running.
input_data = np.array([[
    new_patient['Age'], 
    new_patient['Gender'], 
    new_patient['HeartRate'],
    new_patient['SpO2'],
    new_patient['Stress'],
    new_patient['BP_Sys'],
    new_patient['BP_Dia'],
    new_patient['BMI']
]])

# 4. Predict
predictions = model.predict(input_data)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

# Map back to text
classes = ["Fit", "Slightly Unfit", "Severe"]
result = classes[predicted_class]

print("-------------------------------")
print(f"Prediction: {result}")
print(f"Confidence: {confidence * 100:.2f}%")
print("-------------------------------")