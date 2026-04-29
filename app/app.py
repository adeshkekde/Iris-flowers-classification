from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("models/model.pkl", "rb") as f:
    model, scaler, encoder = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

# 🔥 API route (for modern UI)
@app.route("/predict_api", methods=["POST"])
def predict_api():
    values = request.json["values"]
    data = np.array([values])

    # Scale
    data = scaler.transform(data)

    # Predict
    prediction = model.predict(data)

    # Convert to species
    result = encoder.inverse_transform(prediction)[0]

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
