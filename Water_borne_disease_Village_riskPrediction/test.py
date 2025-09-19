import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load test data
test_df = pd.read_csv("Data/water_risk_data_1000.csv")

# 2. Features & labels
features = [
    "turbidity", "pH", "TDS", "color_smell",
    "cases_per_1000", "abd_pain_per_1000", "fever_per_1000", "dehydr_per_1000"
]
X_test = test_df[features]

# 3. Load trained model and risk labels
model = joblib.load("xgboost_WBD.pkl")
risk_labels = joblib.load("risk_labels.pkl")  # Example: ["Low","Guarded","Moderate","High","Very High"]

# 4. Encode test labels according to saved order
y_test = pd.Categorical(test_df["risk_category"], categories=risk_labels, ordered=True).codes

# 5. Make predictions
y_pred = model.predict(X_test)  # Categorical prediction
y_pred_proba = model.predict_proba(X_test)  # Probability for each risk level

# 6. Decode predictions back to category names
y_test_labels = [risk_labels[i] for i in y_test]
y_pred_labels = [risk_labels[i] for i in y_pred]

# 7. Add risk scores to the dataframe
test_df["Predicted_Risk_Category"] = y_pred_labels
test_df["Risk_Score"] = np.max(y_pred_proba, axis=1)  # Highest probability as confidence score

# OPTIONAL: Save probabilities for each class
for i, label in enumerate(risk_labels):
    test_df[f"Prob_{label}"] = y_pred_proba[:, i]

# Save results
test_df.to_csv("water_risk_predictions.csv", index=False)

# 8. Evaluate model
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print(f"✅ Test Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=risk_labels))

# 9. Confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=risk_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=risk_labels, yticklabels=risk_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - XGBoost (Waterborne Risk)")
plt.show()

print("\nSample Output with Risk Scores:")
print(test_df[["turbidity", "pH", "Predicted_Risk_Category", "Risk_Score"]].head(10))
