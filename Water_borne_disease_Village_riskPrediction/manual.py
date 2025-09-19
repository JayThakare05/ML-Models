import joblib
import pandas as pd

# ===== 1. Load the trained model =====
model_filename = "xgboost_WBD.pkl"  # Make sure this matches your saved model
model = joblib.load(model_filename)

# ===== 2. Manual input values (Example for a particular area) =====
manual_input = {
    "turbidity": 8.5,            # NTU
    "pH": 6.9,                    # pH level
    "TDS": 780.5,                 # Total dissolved solids (mg/L)
    "color_smell": 1,             # 1 = Unsafe color/smell, 0 = Safe
    "cases_per_1000": 15,         # Total cases per 1000 people
    "abd_pain_per_1000": 6,       # Abdominal pain per 1000
    "fever_per_1000": 8,          # Fever per 1000
    "dehydr_per_1000": 3          # Dehydration per 1000
}

# Convert to dataframe for model prediction
input_df = pd.DataFrame([manual_input])

# ===== 3. Predict risk category from trained ML model =====
predicted_category = model.predict(input_df)[0]

# ===== 4. Custom formula to calculate risk score =====
def calculate_risk_score(data):
    # --- Individual scoring functions ---
    def turbidity_score(val): return 0 if val <= 1 else (40 if val <= 5 else (70 if val <= 10 else 100))
    def ph_score(val): return 0 if 6.5 <= val <= 8.5 else (40 if 6.0 <= val <= 9.0 else 80)
    def tds_score(val): return 0 if val <= 500 else (40 if val <= 1000 else (70 if val <= 1500 else 100))
    def color_score(val): return val * 30  # Unsafe water gives 30 points
    def cases_score(val): return min(val * 4, 100)  # Cap at 100
    def abd_score(val): return min(val * 6, 100)
    def fever_score(val): return min(val * 5, 100)
    def dehydr_score(val): return min(val * 10, 100)

    # --- Weights for each parameter ---
    weights = {
        "turbidity": 0.20,
        "pH": 0.10,
        "TDS": 0.10,
        "color": 0.10,
        "cases": 0.20,
        "abd": 0.05,
        "fever": 0.10,
        "dehydr": 0.15
    }

    # --- Weighted score calculation ---
    score = (
        turbidity_score(data["turbidity"]) * weights["turbidity"] +
        ph_score(data["pH"]) * weights["pH"] +
        tds_score(data["TDS"]) * weights["TDS"] +
        color_score(data["color_smell"]) * weights["color"] +
        cases_score(data["cases_per_1000"]) * weights["cases"] +
        abd_score(data["abd_pain_per_1000"]) * weights["abd"] +
        fever_score(data["fever_per_1000"]) * weights["fever"] +
        dehydr_score(data["dehydr_per_1000"]) * weights["dehydr"]
    )
    return round(score, 2)

# Calculate custom risk score
risk_score = calculate_risk_score(manual_input)

# ===== 5. Map custom risk score to risk category =====
def map_score_to_category(score):
    if score <= 25:
        return "Low"
    elif score <= 50:
        return "Moderate"
    elif score <= 75:
        return "High"
    return "Very High"

calculated_risk_category = map_score_to_category(risk_score)

# ===== 6. Print the results =====
print("\n===== Waterborne Disease Risk Prediction =====")
print("Manual Input Data:", manual_input)
print(f"\nPredicted Risk Category (ML Model): {predicted_category}")
print(f"Calculated Risk Score (Custom Formula): {risk_score}")
print(f"Calculated Risk Category (Custom Formula): {calculated_risk_category}")
