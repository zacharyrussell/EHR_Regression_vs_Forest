import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
patients_df = pd.read_csv("./csv/patients.csv")
conditions_df = pd.read_csv("./csv/conditions.csv")
encounters_df = pd.read_csv("./csv/encounters.csv") 
observations_df = pd.read_csv("./csv/observations.csv") 

# Append AGE field
patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'])
patients_df['AGE'] = patients_df['BIRTHDATE'].apply(lambda x: datetime.now().year - x.year)

# Convert gender to numerical 
patients_df['GENDER'] = patients_df['GENDER'].map({'M' : 0, 'F': 1})
# Create Diabetes field
conditions_df['Diabetes'] = conditions_df['DESCRIPTION'].apply(lambda x: 1 if 'diabetes' in str(x).lower() else 0)
data = conditions_df.merge(patients_df, left_on="PATIENT", right_on="Id", how='left')

# Count number of conditions per patient
data['Condition_Count'] = data.groupby('PATIENT')['DESCRIPTION'].transform('count')

# Hospital visits per patient
encounter_counts = encounters_df.groupby("PATIENT").size().reset_index(name="Hospital_Visits")
data = data.merge(encounter_counts, on="PATIENT", how='left')
data['Hospital_Visits'] = data['Hospital_Visits'].fillna(0)

# Create BMI field
bmi_data = observations_df[observations_df['DESCRIPTION'].str.contains('BMI', case=False, na=False)].copy()
bmi_data.loc[:, 'VALUE'] = pd.to_numeric(bmi_data['VALUE'], errors='coerce')
bmi_data = bmi_data.dropna(subset=['VALUE'])
# Compute average BMI per patient
bmi_data = bmi_data.groupby('PATIENT')['VALUE'].mean().reset_index(name="BMI")
data = data.merge(bmi_data, on="PATIENT", how="left")

# features as predictors
features = ['AGE', 'GENDER', 'Condition_Count', 'Hospital_Visits', 'BMI']
target = 'Diabetes'
data = data.drop_duplicates(subset=['PATIENT'])


# Create test/training sets
X = data[features]
y = data[target]

X = X.fillna(X.median())
# 80:20 split for training/testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Analyze Data Distribution
print("Class Distribution:")
print(np.bincount(y_train))
print(np.bincount(y_test))



# REGRESSION PREDICTION
model = LogisticRegression(class_weight='balanced', max_iter=500)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# Evaluate Regression
print("Model Accuracy : ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))



# RANDOM FOREST
# Compare to results of random forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=1))