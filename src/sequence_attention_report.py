
import numpy as np
import pandas as pd
import os

# Original sequence
sequence = "ACGUACGUACGGCUA"

# Load attention weights
weights = np.loadtxt(
    "outputs/reports/attention_weights.csv",
    delimiter=","
) if os.path.exists(
    "outputs/reports/attention_weights.csv"
) else None

# Recreate weights from the latest sequence implementation
if weights is None or weights.shape != (len(sequence), len(sequence)):
    print("Please first save the latest sequence attention weights.")
    raise SystemExit

# Create report
report = {
    "Sequence": sequence,
    "Sequence_Length": len(sequence),
    "Nucleotide_Types": len(set(sequence)),
    "Attention_Matrix_Rows": weights.shape[0],
    "Attention_Matrix_Columns": weights.shape[1],
    "Average_Attention_Weight": weights.mean(),
    "Maximum_Attention_Weight": weights.max(),
    "Minimum_Attention_Weight": weights.min(),
    "All_Rows_Normalized": np.allclose(
        weights.sum(axis=1), 1.0
    )
}

os.makedirs("outputs/reports", exist_ok=True)

pd.DataFrame([report]).to_csv(
    "outputs/reports/sequence_attention_summary.csv",
    index=False
)

print("SEQUENCE ATTENTION REPORT")
for key, value in report.items():
    print(f"{key}: {value}")

print("\nReport saved successfully!")
