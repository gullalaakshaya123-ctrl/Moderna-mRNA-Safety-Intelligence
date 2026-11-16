
import pandas as pd

# Load original dataset
df = pd.read_csv("data/patient_records (1).csv")

# Validate required columns
required_columns = [
    "Patient_ID",
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History",
    "Reaction_Score"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Fill numerical missing values using median
df["Biomarker_Level"] = df["Biomarker_Level"].fillna(
    df["Biomarker_Level"].median()
)

# Fill categorical missing values using mode
history_mode = df["Prior_Reaction_History"].mode()[0]

df["Prior_Reaction_History"] = (
    df["Prior_Reaction_History"].fillna(history_mode)
)

# Verify cleaned dataset
print("Missing values after cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("data/patient_records_clean.csv", index=False)

print("\nDataset shape:", df.shape)
print("Reaction history imputed using mode:", history_mode)
print("Total missing values:", df.isnull().sum().sum())
print("Cleaned dataset saved successfully!")
