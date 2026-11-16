
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Create output directories
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/patient_records_clean.csv")

print("Dataset loaded:", df.shape)

# Separate features and target
X = df.drop(columns=["Patient_ID", "Reaction_Score"])
y = df["Reaction_Score"]

# Feature groups
numerical_features = [
    "Age",
    "Biomarker_Level",
    "Vaccine_Dose_mcg",
    "Pre_Existing_Conditions"
]

categorical_features = ["Prior_Reaction_History"]

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )
}

results = []
trained_models = {}
roc_results = []

# Train and evaluate models
for name, classifier in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    predictions = pipeline.predict(X_test)

    # Prediction probabilities for ROC
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)

    roc_results.append({
        "Model": name,
        "FPR": fpr,
        "TPR": tpr,
        "AUC": auc_score
    })

    # Calculate evaluation metrics
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": auc_score
    }

    results.append(metrics)
    trained_models[name] = pipeline

    # Save classification report
    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    report_path = (
        f"outputs/reports/{name.lower().replace(' ', '_')}_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(report)

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Reaction", "Reaction"],
        yticklabels=["No Reaction", "Reaction"]
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    cm_path = (
        f"outputs/figures/{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(cm_path, dpi=300)
    plt.close()

    print("\n", name)
    print("Accuracy:", round(metrics["Accuracy"], 4))
    print("Precision:", round(metrics["Precision"], 4))
    print("Recall:", round(metrics["Recall"], 4))
    print("F1 Score:", round(metrics["F1 Score"], 4))
    print("ROC AUC:", round(auc_score, 4))

# Save model comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/reports/model_comparison.csv",
    index=False
)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# Plot ROC curves
plt.figure(figsize=(8, 6))

for result in roc_results:
    plt.plot(
        result["FPR"],
        result["TPR"],
        label=f'{result["Model"]} (AUC = {result["AUC"]:.3f})'
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/roc_curve.png",
    dpi=300
)

plt.close()

# Save ROC AUC scores
auc_df = pd.DataFrame([
    {
        "Model": result["Model"],
        "ROC AUC": result["AUC"]
    }
    for result in roc_results
])

auc_df.to_csv(
    "outputs/reports/roc_auc_scores.csv",
    index=False
)

# Select best model based on F1 Score
best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

# Save best model
joblib.dump(
    best_model,
    "outputs/risk_model.joblib"
)

# Save best model name
with open("outputs/reports/best_model.txt", "w") as file:
    file.write(f"Best Model: {best_model_name}\n")
    file.write(
        f"F1 Score: {results_df.loc[results_df['Model'] == best_model_name, 'F1 Score'].iloc[0]:.4f}\n"
    )

# Final results
print("\n" + "=" * 45)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 45)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\nBest Model:", best_model_name)
print("Saved Model: outputs/risk_model.joblib")
print("ROC Curve: outputs/figures/roc_curve.png")
print("AUC Scores: outputs/reports/roc_auc_scores.csv")
