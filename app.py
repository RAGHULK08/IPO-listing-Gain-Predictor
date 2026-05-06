"""
IPO Listing Gain Predictor — Flask Backend
==========================================
Serves the trained PyTorch model via a REST API.
Run: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import pickle

app = Flask(__name__, static_folder="static")
CORS(app)  # Allow frontend to call this API

# ── 1. Architecture (must match notebook exactly) ─────────────────────────────
def build_model(input_size=9):
    return nn.Sequential(
        nn.Linear(input_size, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.25),

        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.125),

        nn.Linear(32, 16),
        nn.ReLU(),

        nn.Linear(16, 1),
        nn.Sigmoid()
    )

# ── 2. Load model + scaler at startup ─────────────────────────────────────────
MODEL_PATH  = "ipo_model.pth"
SCALER_PATH = "scaler.pkl"

model  = None
scaler = None

def load_artifacts():
    global model, scaler
    try:
        model = build_model(input_size=9)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        model.eval()
        print(f"[✓] Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] Model file '{MODEL_PATH}' not found — using DEMO mode.")
    except Exception as e:
        print(f"[!] Could not load model: {e} — using DEMO mode.")

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"[✓] Scaler loaded from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"[!] Scaler file '{SCALER_PATH}' not found — using DEMO mode.")

load_artifacts()


# ── 3. Helper: build feature vector from form inputs ──────────────────────────
FEATURE_ORDER = [
    "subscription_qib",   # QIB subscription multiple
    "subscription_hni",   # HNI subscription multiple
    "subscription_rii",   # RII subscription multiple
    "issue_price",        # Issue price (₹)
    "issue_size",         # Issue size (₹ Cr)
    "year",               # Listing year
    "month",              # Listing month (1–12)
    "quarter",            # Listing quarter (1–4)
    "day_of_week_enc",    # Day of week encoded (0=Mon … 4=Fri)
]

import pandas as pd

NOTEBOOK_FEATURE_COLUMNS = [
    "Subscription_QIB", "Subscription_HNI", "Subscription_RII",
    "Issue_Price", "Issue_Size", "Year", "Month", "Quarter", "DayOfWeek_Enc"
]

def build_feature_vector(data: dict) -> pd.DataFrame:
    values = [float(data[f]) for f in FEATURE_ORDER]
    return pd.DataFrame([values], columns=NOTEBOOK_FEATURE_COLUMNS)


# ── 4. API Endpoints ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend HTML."""
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON): { subscription_qib, subscription_hni, subscription_rii,
                   issue_price, issue_size, year, month, quarter, day_of_week_enc }
    Returns:     { probability, prediction, confidence_band }
    """
    try:
        data = request.get_json(force=True)
        features = build_feature_vector(data)

        # ── DEMO mode: simple heuristic when model files are absent ──────────
        if model is None or scaler is None:
            avg_sub = (float(data["subscription_qib"]) +
                       float(data["subscription_hni"]) +
                       float(data["subscription_rii"])) / 3
            prob = float(min(0.95, max(0.05, 0.35 + avg_sub * 0.018)))
        else:
            scaled   = scaler.transform(features)
            tensor   = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                prob = float(model(tensor).squeeze())

        prediction     = "Profitable" if prob > 0.5 else "Not Profitable"
        confidence_band = (
            "Strong Buy"   if prob > 0.75 else
            "Borderline"   if prob > 0.50 else
            "Avoid"        if prob > 0.25 else
            "Strong Avoid"
        )

        return jsonify({
            "probability":      round(prob * 100, 1),
            "prediction":       prediction,
            "confidence_band":  confidence_band,
            "demo_mode":        model is None,
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded":  model  is not None,
        "scaler_loaded": scaler is not None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
