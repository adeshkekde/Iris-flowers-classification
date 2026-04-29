import pickle
import numpy as np

# Load model
with open("models/model.pkl", "rb") as f:
    model, scaler, encoder = pickle.load(f)

# Example input
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Scale
sample = scaler.transform(sample)

# Predict
prediction = model.predict(sample)

# Convert back to label
species = encoder.inverse_transform(prediction)

print("Predicted Species:", species[0])