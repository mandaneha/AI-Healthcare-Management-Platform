"""Synthetic, fully artificial healthcare data. No real patient information."""
import numpy as np
import pandas as pd

from .config import DEPARTMENTS, DOCTORS


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_patients(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(48, 18, n), 1, 95).astype(int)
    gender = rng.choice(["Female", "Male", "Other"], n, p=[0.49, 0.49, 0.02])
    bmi = np.clip(rng.normal(26.5, 5, n), 14, 45).round(1)
    systolic_bp = np.clip(rng.normal(110 + 0.45 * age, 14), 90, 200).round().astype(int)
    glucose = np.clip(rng.normal(90 + 0.5 * age + 1.2 * (bmi - 25), 22), 65, 350).round().astype(int)
    cholesterol = np.clip(rng.normal(170 + 0.6 * age, 30), 100, 350).round().astype(int)
    smoker = (rng.random(n) < np.where(age > 18, 0.18, 0.0)).astype(int)
    num_conditions = np.clip(rng.poisson(0.2 + age / 45 + (bmi > 30) * 0.5), 0, 8)
    prior_admissions = np.clip(rng.poisson(0.2 + 0.01 * age + 0.3 * num_conditions), 0, 10)
    avg_stay_days = np.clip(rng.normal(3 + 0.8 * num_conditions, 1.5), 0, 20).round(1)

    logit = (-5.0 + 0.025 * age + 0.010 * (glucose - 100) + 0.015 * (systolic_bp - 120)
             + 0.45 * num_conditions + 0.4 * prior_admissions + 0.08 * avg_stay_days
             + 0.5 * smoker + 0.03 * (bmi - 25))
    readmitted = (rng.random(n) < _sigmoid(logit)).astype(int)

    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "name": [f"Patient {i:04d}" for i in range(1, n + 1)],
        "age": age, "gender": gender, "bmi": bmi, "systolic_bp": systolic_bp,
        "glucose": glucose, "cholesterol": cholesterol, "smoker": smoker,
        "num_conditions": num_conditions, "prior_admissions": prior_admissions,
        "avg_stay_days": avg_stay_days, "readmitted_30d": readmitted,
    })


def generate_appointments(patients: pd.DataFrame, n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pid = rng.choice(patients["id"].to_numpy(), n)
    ages = patients.set_index("id").loc[pid, "age"].to_numpy()
    dept = rng.choice(DEPARTMENTS, n)
    doctor = [DOCTORS[d] for d in dept]
    offset = rng.integers(-180, 45, n)
    today = pd.Timestamp.today().normalize()
    scheduled = today + pd.to_timedelta(offset, unit="D") + pd.to_timedelta(rng.integers(9, 17, n), unit="h")
    lead_days = rng.integers(0, 45, n)
    reminder = (rng.random(n) < 0.6).astype(int)
    prior_ns = np.clip(rng.poisson(0.4, n), 0, 6)

    p_ns = _sigmoid(-2.2 + 0.035 * lead_days - 0.9 * reminder + 0.45 * prior_ns - 0.01 * (ages - 40))
    no_show = rng.random(n) < p_ns
    status = np.where(offset < 0, np.where(no_show, "No-Show", "Completed"), "Scheduled")

    return pd.DataFrame({
        "id": np.arange(1, n + 1), "patient_id": pid, "department": dept, "doctor": doctor,
        "scheduled_for": scheduled, "lead_days": lead_days, "reminder_sent": reminder,
        "prior_no_shows": prior_ns, "status": status,
    })
