"""Retrain both models from the current database:  python -m scripts.train"""
from src import db
from src.models import train_noshow_model, train_risk_model

if __name__ == "__main__":
    db.init_db()
    db.seed_if_empty()
    p, a = db.read_patients(), db.read_appointments()
    print("Readmission risk :", train_risk_model(p)["metrics"])
    print("No-show          :", train_noshow_model(a, p)["metrics"])
