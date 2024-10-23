# Code_Alpha_Disease-Prediction-from-Medical-Dat(heart disease)
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import files

# Step 2: Upload the dataset
uploaded = files.upload()

# Replace 'simulated_medical_data.csv' with the file name after uploading
df = pd.read_csv('/content/simulated_medical_data.csv')

# Step 3: Data Exploration
print(df.head())  # Display the first few rows
print(df.info())  # Get information about the dataset

# Step 4: Feature Selection and Target Variable
X = df.drop('disease', axis=1)  # Features (everything except 'disease')
y = df['disease']               # Target variable (0: no disease, 1: has disease)

# Step 5: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale the Features (important for many ML algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Step 8: Make Predictions and Evaluate the Model
y_pred = clf.predict(X_test_scaled)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Calculate Disease Risk (Probabilities) for Test Data
y_prob = clf.predict_proba(X_test_scaled)  # Probabilities for each class

# The second column (index 1) contains the probability of class '1' (disease)
risk_of_disease = y_prob[:, 1]

# Step 10: Combine the risk with patient data for analysis
df_test = X_test.copy()  # Create a copy of the test features
df_test['Actual Disease'] = y_test.values  # Add the actual disease label
df_test['Predicted Disease Risk'] = risk_of_disease  # Add the predicted risk (probability)

# Show the first 10 records with disease risk
print(df_test.head(10))

# Step 11: Categorize the Risk into Low, Moderate, and High Risk
def categorize_risk(probability):
    if probability >= 0.8:
        return 'High Risk'
    elif probability >= 0.5:
        return 'Moderate Risk'
    else:
        return 'Low Risk'

df_test['Risk Category'] = df_test['Predicted Disease Risk'].apply(categorize_risk)

# Show the first 10 records with risk categories
print(df_test[['age', 'cholesterol', 'Predicted Disease Risk', 'Risk Category']].head(10))


