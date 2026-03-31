from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained ML model (HistGradientBoostingClassifier)
# Trained on creditcard_2023.csv with features: V1-V28 + Amount (29 features)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_ml_model.joblib")
ml_model = None

try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"✓ ML model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Could not load ML model: {e}")


def map_inputs_to_features(features: dict) -> np.ndarray:
    """
    Maps human-readable transaction inputs into a 29-feature vector (V1–V28, Amount)
    that the trained model expects.

    The V-features in the original dataset are PCA-transformed components.
    We approximate meaningful signals by mapping user inputs to the most
    important V-features identified during EDA / feature selection:
        V14 (21.4%), V10 (15.2%), V4 (9.7%), V12 (9.3%),
        V11 (8.6%), V17 (8.3%), V16 (7.2%), V7 (4.4%), V3 (3.0%)
    """
    amount = float(features.get("amount", 0))
    hour   = int(features.get("hour", 12))
    distance = float(features.get("distance_from_home", 0))
    online = 1.0 if str(features.get("online_transaction", "no")).lower() == "yes" else 0.0
    new_merchant = 1.0 if str(features.get("new_merchant", "no")).lower() == "yes" else 0.0
    prev_fraud = int(features.get("previous_frauds", 0))
    card_present = 1.0 if str(features.get("card_present", "yes")).lower() == "yes" else 0.0
    country_mismatch = 1.0 if str(features.get("country_mismatch", "no")).lower() == "yes" else 0.0

    # Start with zeros for all 28 V-features
    V = np.zeros(28)

    # --- Map user signals to the most discriminative V-features ---
    # V14 (strongest fraud indicator): negative values = fraud pattern
    if amount > 5000:
        V[13] = -3.5
    elif amount > 2000:
        V[13] = -1.5
    else:
        V[13] = 1.0

    # V10: negative = fraud
    if hour < 5 or hour > 23:
        V[9] = -3.0
    else:
        V[9] = 0.5

    # V4: positive = fraud
    if distance > 500:
        V[3] = 4.0
    elif distance > 100:
        V[3] = 2.0
    else:
        V[3] = 0.2

    # V12: negative = fraud
    if new_merchant:
        V[11] = -4.0
    else:
        V[11] = 0.5

    # V11: positive = fraud
    if online:
        V[10] = 3.5
    else:
        V[10] = -0.2

    # V17: negative = fraud
    if prev_fraud > 0:
        V[16] = -3.0 * min(prev_fraud, 3)
    else:
        V[16] = 0.3

    # V16: negative = fraud
    if not card_present:
        V[15] = -3.0
    else:
        V[15] = 0.3

    # V7: positive = fraud
    if country_mismatch:
        V[6] = 4.0
    else:
        V[6] = 0.1

    # V3: positive for normal, negative for fraud
    if amount < 500 and card_present and not online:
        V[2] = 1.5
    elif amount > 3000 and not card_present:
        V[2] = -2.0

    # Build the full 29-feature vector: [V1...V28, Amount]
    feature_vector = np.append(V, amount).reshape(1, -1)
    return feature_vector


def build_flags(features: dict) -> list:
    """Generate human-readable risk flags from transaction inputs."""
    flags = []
    amount = float(features.get("amount", 0))
    hour = int(features.get("hour", 12))
    distance = float(features.get("distance_from_home", 0))
    online = str(features.get("online_transaction", "no")).lower() == "yes"
    new_merchant = str(features.get("new_merchant", "no")).lower() == "yes"
    prev_fraud = int(features.get("previous_frauds", 0))
    card_present = str(features.get("card_present", "yes")).lower() == "yes"
    country_mismatch = str(features.get("country_mismatch", "no")).lower() == "yes"

    if amount > 5000:
        flags.append("High transaction amount (>$5,000)")
    elif amount > 2000:
        flags.append("Elevated transaction amount (>$2,000)")
    if hour < 5 or hour > 23:
        flags.append("Transaction at unusual hour")
    if distance > 500:
        flags.append("Transaction far from home location")
    elif distance > 100:
        flags.append("Elevated distance from home")
    if online:
        flags.append("Online / card-not-present channel")
    if new_merchant:
        flags.append("First-time merchant")
    if prev_fraud > 0:
        flags.append(f"{prev_fraud} prior fraud incident(s) on account")
    if not card_present:
        flags.append("Physical card not present")
    if country_mismatch:
        flags.append("Country mismatch (IP vs billing)")

    return flags if flags else ["No suspicious signals detected"]


def predict(features: dict) -> dict:
    """
    Uses the trained ML model to predict fraud.
    Falls back to heuristic scoring if model is unavailable.
    Returns {label, confidence, risk_score, flags}
    """
    flags = build_flags(features)

    if ml_model is not None:
        # ----- ML Model Prediction -----
        feature_vector = map_inputs_to_features(features)
        prob = ml_model.predict_proba(feature_vector)[0][1]  # P(fraud)
        risk_score = int(round(prob * 100))
        label = "Fraudulent" if prob >= 0.5 else "Legitimate"
        confidence = round(prob * 100, 1) if label == "Fraudulent" else round((1 - prob) * 100, 1)
    else:
        # ----- Fallback: Heuristic scoring -----
        risk_score = 0
        amount = float(features.get("amount", 0))
        if amount > 5000: risk_score += 25
        elif amount > 2000: risk_score += 10
        hour = int(features.get("hour", 12))
        if hour < 5 or hour > 23: risk_score += 20
        distance = float(features.get("distance_from_home", 0))
        if distance > 500: risk_score += 20
        elif distance > 100: risk_score += 8
        if str(features.get("online_transaction", "no")).lower() == "yes": risk_score += 10
        if str(features.get("new_merchant", "no")).lower() == "yes": risk_score += 15
        prev = int(features.get("previous_frauds", 0))
        if prev > 0: risk_score += min(prev * 15, 30)
        if str(features.get("card_present", "yes")).lower() != "yes": risk_score += 12
        if str(features.get("country_mismatch", "no")).lower() == "yes": risk_score += 25
        risk_score = min(risk_score, 100)
        label = "Fraudulent" if risk_score >= 45 else "Legitimate"
        confidence = round(risk_score if label == "Fraudulent" else (100 - risk_score), 1)

    return {
        "label": label,
        "confidence": confidence,
        "risk_score": risk_score,
        "flags": flags,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    result = predict(data)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)