
import numpy as np
import pandas as pd

# Load saved attention weights
weights = np.loadtxt(
    "outputs/reports/attention_weights.csv",
    delimiter=","
)

print("SELF-ATTENTION ANALYSIS")

print("\nAttention Weight Matrix:")
print(weights)

# Calculate row sums
row_sums = np.sum(weights, axis=1)

print("\nAttention Row Sums:")
print(row_sums)

# Find the highest attention position for each query
highest_positions = np.argmax(weights, axis=1)

print("\nHighest Attention Position:")
for i, position in enumerate(highest_positions):
    print(f"Query {i} focuses most on Key {position}")

# Create attention analysis table
analysis = pd.DataFrame({
    "Query_Position": range(len(weights)),
    "Highest_Attention_Key": highest_positions,
    "Maximum_Attention_Weight": np.max(weights, axis=1),
    "Row_Sum": row_sums
})

analysis.to_csv(
    "outputs/reports/attention_analysis.csv",
    index=False
)

print("\nAttention analysis saved successfully!")

# Verify normalization
if np.allclose(row_sums, 1.0):
    print("PASS: All attention rows sum to 1.")
else:
    print("FAIL: Attention normalization issue detected.")
