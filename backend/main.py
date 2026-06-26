"""
Fraud Detection API - serves the trained XGBoost credit-card fraud model.

This replaces the old Streamlit app's in-page scoring, and fixes the bug where
the scorecard fed the model partly-zeroed input: it built a dict with keys like
`trans_day` / `hours_since_last_trans` that aren't in the model's 31-feature
schema (the real ones are `trans_day_of_week`, `city_pop`, `unix_time`,
`category_avg_amt`), so those features silently stayed 0 on every prediction.
Here the full 31-feature vector is constructed server-side from the real schema.
"""
import os
import pickle
from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")


def _load():
    with open(os.path.join(MODELS, "fraud_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS, "features.pkl"), "rb") as f:
        features = list(pickle.load(f))
    with open(os.path.join(MODELS, "threshold.pkl"), "rb") as f:
        threshold = float(pickle.load(f))
    return model, features, threshold


model, FEATURES, THRESHOLD = _load()
# Categories the model was one-hot encoded on (drives the UI dropdown).
CATEGORIES = [f[len("cat_"):] for f in FEATURES if f.startswith("cat_")]

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost credit-card fraud scorer (31 engineered features, tuned threshold).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreRequest(BaseModel):
    amount: float = 100.0
    category: str = "shopping_net"
    hour: int = 14
    day_of_week: int = 2  # 0 = Monday ... 6 = Sunday
    distance_from_home: float = 5.0
    age: int = 35
    gender: str = "Male"
    customer_avg_amt: float = 50.0
    merchant_risk: float = 0.5
    city_pop: int = 50_000
    is_new_merchant: bool = False
    merchant_visit_count: int = 5


def build_vector(r: ScoreRequest) -> pd.DataFrame:
    """Construct the full, correctly-named 31-feature row (no silent zeros)."""
    now = datetime.now()
    amt = float(r.amount)
    avg = float(r.customer_avg_amt)
    hour = max(0, min(23, int(r.hour)))
    dow = max(0, min(6, int(r.day_of_week)))

    values = {
        "amt": amt,
        "city_pop": float(r.city_pop),
        # representative timestamp from today's date at the chosen hour
        "unix_time": float(datetime(now.year, now.month, now.day, hour).timestamp()),
        "trans_hour": hour,
        "trans_day_of_week": dow,
        "is_weekend": 1 if dow >= 5 else 0,
        "trans_month": now.month,
        "distance_from_home": float(r.distance_from_home),
        "customer_avg_amt": avg,
        "amt_deviation": amt - avg,
        "amt_ratio": amt / (avg + 0.01),
        "category_avg_amt": avg,  # proxy: customer average as the category baseline
        "merchant_visit_count": int(r.merchant_visit_count),
        "is_new_merchant": 1 if r.is_new_merchant else 0,
        "age": int(r.age),
        "gender_encoded": 1 if r.gender.lower().startswith("m") else 0,
        "merchant_encoded": float(r.merchant_risk),
    }

    row = {f: 0.0 for f in FEATURES}
    for k, v in values.items():
        if k in row:
            row[k] = v
    cat_key = f"cat_{r.category}"
    if cat_key in row:
        row[cat_key] = 1.0
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)


def risk_tier(p: float) -> str:
    if p < THRESHOLD * 0.4:
        return "LOW"
    if p < THRESHOLD:
        return "MEDIUM"
    if p < THRESHOLD + 0.25:
        return "HIGH"
    return "CRITICAL"


@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/metadata", "/api/score"],
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metadata")
def metadata():
    """Threshold, schema, categories, and REAL feature importances from the model."""
    importance = {}
    fi = getattr(model, "feature_importances_", None)
    if fi is not None:
        importance = {f: float(w) for f, w in zip(FEATURES, fi)}
    return {
        "threshold": THRESHOLD,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "categories": CATEGORIES,
        "feature_importance": importance,
    }


@app.post("/api/score")
def score(req: ScoreRequest):
    fv = build_vector(req)
    prob = float(model.predict_proba(fv)[0][1])
    ratio = req.amount / (req.customer_avg_amt + 0.01)
    factors = [
        {"factor": "Amount vs. average", "value": f"${req.amount:,.0f} · {ratio:.1f}x avg", "signal": "high" if ratio >= 2 else "normal"},
        {"factor": "Distance from home", "value": f"{req.distance_from_home:,.0f} mi", "signal": "high" if req.distance_from_home > 50 else "normal"},
        {"factor": "Transaction hour", "value": f"{req.hour:02d}:00", "signal": "high" if req.hour < 6 else "normal"},
        {"factor": "Customer age", "value": str(req.age), "signal": "high" if (req.age < 18 or req.age > 80) else "normal"},
        {"factor": "Merchant risk", "value": f"{req.merchant_risk:.2f}", "signal": "high" if req.merchant_risk > 0.6 else "medium" if req.merchant_risk > 0.4 else "normal"},
    ]
    return {
        "probability": prob,
        "threshold": THRESHOLD,
        "is_fraud": bool(prob >= THRESHOLD),
        "risk_tier": risk_tier(prob),
        "confidence": max(prob, 1 - prob),
        "factors": factors,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
