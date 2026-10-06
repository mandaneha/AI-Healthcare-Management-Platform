"""First-run setup: create DB, seed synthetic data, train models if missing."""
from . import db
from .config import NOSHOW_MODEL_PATH, RISK_MODEL_PATH
from .models import train_noshow_model, train_risk_model


def bootstrap() -> None:
    db.init_db()
    db.seed_if_empty()
    if not RISK_MODEL_PATH.exists():
        train_risk_model(db.read_patients())
    if not NOSHOW_MODEL_PATH.exists():
        train_noshow_model(db.read_appointments(), db.read_patients())


if __name__ == "__main__":
    bootstrap()
    print("Database seeded and models trained.")
