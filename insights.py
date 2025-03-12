import pandas as pd

# Load patient data
patients_df = pd.read_csv("./csv/patients.csv")
conditions_df = pd.read_csv("./csv/conditions.csv")
medications_df = pd.read_csv("./csv/medications.csv")

print(patients_df.head())



import matplotlib.pyplot as plt 

gender_counts = patients_df["GENDER"].value_counts()
gender_counts.plot(kind='bar', title='Patient Gender Distribution')
plt.xlabel("Gender")
plt.ylabel("Count")



top_conditions = conditions_df['DESCRIPTION'].value_counts().head(10)

top_conditions.plot(kind="bar", title="Top diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()