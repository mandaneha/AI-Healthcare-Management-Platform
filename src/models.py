"""Training and inference for the two ML models.

1. Readmission risk  - XGBoost classifier on patient clinical features.
2. Appointment no-show - Gradient boosting pipeline on scheduling features.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import NOSHOW_MODEL_PATH, RISK_MODEL_PATH

RISK_FEATURES = ["age", "bmi", "systolic_bp", "glucose", "cholesterol", "smoker",
                 "num_conditions", "prior_admissions", "avg_stay_days"]
NOSHOW_FEATURES = ["lead_days", "reminder_sent", "prior_no_shows", "age", "department", "weekday"]


def risk_level(p: float, low: float = 0.15, high: float = 0.35) -> str:
    return "Low" if p < low else "Medium" if p < high else "High"


def train_risk_model(patients: pd.DataFrame) -> dict:
    from xgboost import XGBClassifier  # imported lazily so the app still opens without it

    X, y = patients[RISK_FEATURES], patients["readmitted_30d"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9,
                          colsample_bytree=0.9, eval_metric="logloss", random_state=42)
    model.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    bundle = {"model": model, "features": RISK_FEATURES,
              "metrics": {"roc_auc": round(auc, 3), "n_train": len(X_tr), "positive_rate": round(float(y.mean()), 3)},
              "importances": dict(zip(RISK_FEATURES, map(float, model.feature_importances_)))}
    joblib.dump(bundle, RISK_MODEL_PATH)
    return bundle


def _noshow_frame(appts: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    df = appts.merge(patients[["id", "age"]].rename(columns={"id": "patient_id"}), on="patient_id", how="left")
    df["weekday"] = pd.to_datetime(df["scheduled_for"]).dt.dayofweek
    return df


def train_noshow_model(appts: pd.DataFrame, patients: pd.DataFrame) -> dict:
    df = _noshow_frame(appts, patients)
    df = df[df["status"].isin(["Completed", "No-Show"])]
    X, y = df[NOSHOW_FEATURES], (df["status"] == "No-Show").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pre = ColumnTransformer([("dept", OneHotEncoder(handle_unknown="ignore"), ["department"])],
                            remainder="passthrough")
    pipe = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=42))])
    pipe.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]))
    bundle = {"model": pipe, "features": NOSHOW_FEATURES,
              "metrics": {"roc_auc": round(auc, 3), "n_train": len(X_tr), "positive_rate": round(float(y.mean()), 3)}}
    joblib.dump(bundle, NOSHOW_MODEL_PATH)
    return bundle


def load_bundle(path):
    return joblib.load(path) if path.exists() else None


def predict_risk(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(rows[bundle["features"]])[:, 1]


def predict_noshow(bundle: dict, appts: pd.DataFrame, patients: pd.DataFrame) -> np.ndarray:
    df = _noshow_frame(appts, patients)
    return bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
