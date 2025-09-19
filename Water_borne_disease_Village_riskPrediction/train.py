import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib  # for saving the model

# 1. Load training data
train_df = pd.read_csv("Data/water_risk_data_4000.csv")

# 2. Features & labels
features = [
    "turbidity", "pH", "TDS", "color_smell",
    "cases_per_1000", "abd_pain_per_1000", "fever_per_1000", "dehydr_per_1000"
]
X = train_df[features]

# 3. Define risk categories in correct order
risk_order = ["Low", "Moderate", "High", "Very High"]

# Convert to categorical with fixed order
y = pd.Categorical(train_df["risk_category"], categories=risk_order, ordered=True).codes

# 4. Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# 6. Save the trained model and the category order
joblib.dump(model, "xgboost_WBD.pkl")
joblib.dump(risk_order, "risk_labels.pkl")  # save category mapping

print("✅ Model trained and saved as xgboost_WBD.pkl")
print("✅ Risk category mapping saved as risk_labels.pkl")
