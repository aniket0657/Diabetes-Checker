from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Initialize the Flask application
application = Flask(__name__)
app = application

# Load the scaler and trained model
SCALER_PATH = os.path.join("Model", "standardScalar.pkl")
MODEL_PATH = os.path.join("Model", "modelForPrediction.pkl")

try:
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    raise RuntimeError(f" Failed to load model or scaler: {e}")

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for single prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""
    if request.method == 'POST':
        try:
            # Collect input data from form
            input_features = [
                int(request.form.get("Pregnancies")),
                float(request.form.get("Glucose")),
                float(request.form.get("BloodPressure")),
                float(request.form.get("SkinThickness")),
                float(request.form.get("Insulin")),
                float(request.form.get("BMI")),
                float(request.form.get("DiabetesPedigreeFunction")),
                float(request.form.get("Age")),
            ]

            # Transform the input
            transformed_data = scaler.transform([input_features])

            # Predict using the loaded model
            prediction = model.predict(transformed_data)

            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        except Exception as e:
            result = f"Prediction failed: {e}"

        return render_template('single_prediction.html', result=result)

    return render_template('home.html')


if __name__ == "__main__":
    # For Docker deployment
    app.run(host="0.0.0.0", port=8000, debug=True)
