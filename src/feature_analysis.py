
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Get feature names after preprocessing
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

# Get feature importance
importance = model.named_steps["classifier"].feature_importances_

# Create comparison dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE ANALYSIS")
print(importance_df.to_string(index=False))

# Save results
importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

# Plot feature importance
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("reports/feature_importance.png")
plt.show()

print("\nFeature analysis completed successfully!")
