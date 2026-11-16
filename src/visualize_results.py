
import pandas as pd
import matplotlib.pyplot as plt

# Load model comparison results
df = pd.read_csv("reports/model_comparison.csv")

# Display comparison
print(df)

# Plot accuracy comparison
plt.figure(figsize=(8, 5))

plt.bar(df["Model"], df["Accuracy"])

plt.title("ML Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")

plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

# Save graph
plt.savefig("reports/model_accuracy.png")

plt.show()

print("Accuracy graph saved successfully!")
