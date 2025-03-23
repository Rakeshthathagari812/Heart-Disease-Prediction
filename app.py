#T.RakeshReddy
# MLRIT 22R21A0558 CSE
from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load("heart_disease_model.pkl")  # Loading trained model and preprocessors.
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
                                                  # Defining categorical and numeric features
categorical_features = [
    "sex", "chest_pain_type", "fasting_blood_sugar", "rest_ecg",
    "exercise_induced_angina", "slope", "vessels_colored_by_flourosopy", "thalassemia"
]
numeric_features = ["age", "resting_blood_pressure", "cholestoral", "max_heart_rate", "oldpeak"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        # Get form data
        user_input = {key: request.form[key] for key in request.form}

        # Convert numeric values
        for feature in numeric_features:
            user_input[feature] = float(user_input[feature]) if user_input[feature] else 0.0

        # Encode categorical features
        for feature in categorical_features:
            if feature in label_encoders:
                if user_input[feature] in label_encoders[feature].classes_:
                    user_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
                else:
                    # Assign the first available class if input is unknown
                    user_input[feature] = label_encoders[feature].transform([label_encoders[feature].classes_[0]])[0]

        # Convert input to NumPy array & scale
        input_array = np.array(list(user_input.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        result = " Heart Disease Detected 😞" if prediction == 1 else "No Heart Disease 😊"

        return render_template('results.html', prediction=result)  # Fixed passing variable name


if __name__ == '__main__':
    app.run(debug=True)
