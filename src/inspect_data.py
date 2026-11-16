
import pandas as pd

# Load the patient dataset
file_path = "data/patient_records (1).csv"
df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 patient records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nReaction Score distribution:")
print(df["Reaction_Score"].value_counts())

print("\n========== INSPECTION COMPLETED ==========")
print("\nPrior Reaction History Categories:")
print(df["Prior_Reaction_History"].value_counts(dropna=False))