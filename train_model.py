import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("heart_disease.csv") 

# Display first few rows
print(df.head())

# Convert categorical features to numerical
label_encoders = {}
categorical_columns = ["sex", "chest_pain_type", "fasting_blood_sugar", "rest_ecg", 
                        "exercise_induced_angina", "slope", "vessels_colored_by_flourosopy", "thalassemia"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  

# Define features (X) and target (y)
X = df.drop(columns=["target"])
y = df["target"]

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and encoders
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model saved successfully!")
