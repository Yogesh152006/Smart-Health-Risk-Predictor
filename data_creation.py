import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

# Generate features
age = np.random.randint(20, 80, n)
bmi = np.random.normal(25, 5, n)
blood_pressure = np.random.normal(120, 15, n)
glucose = np.random.normal(100, 30, n)
cholesterol = np.random.normal(200, 40, n)

# Risk logic (simple rule-based probability)
risk_score = (
    (age > 50).astype(int) +
    (bmi > 30).astype(int) +
    (blood_pressure > 140).astype(int) +
    (glucose > 140).astype(int) +
    (cholesterol > 240).astype(int)
)

risk = (risk_score >= 2).astype(int)

# Disease type logic
disease_type = []
for i in range(n):
    if glucose[i] > 140:
        disease_type.append("diabetes")
    elif cholesterol[i] > 240:
        disease_type.append("heart")
    else:
        disease_type.append("none")

# Severity logic
severity = []
for score in risk_score:
    if score <= 1:
        severity.append("low")
    elif score == 2:
        severity.append("medium")
    else:
        severity.append("high")

# Create dataframe
df = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "blood_pressure": blood_pressure,
    "glucose": glucose,
    "cholesterol": cholesterol,
    "risk": risk,
    "disease_type": disease_type,
    "severity": severity
})

df.to_csv("patient_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
