"""Central configuration. Reads optional values from a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'healthcare.db').as_posix()}")
try:
    import streamlit as st
except ImportError:
    st = None

if st is not None:
    GEMINI_API_KEY = str(
        st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    ).strip()

    GEMINI_MODEL = str(
        st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    ).strip()
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

RISK_MODEL_PATH = MODEL_DIR / "readmission_risk.joblib"
NOSHOW_MODEL_PATH = MODEL_DIR / "no_show.joblib"

DEPARTMENTS = [
    "General Medicine", "Cardiology", "Endocrinology",
    "Orthopedics", "Pediatrics", "Dermatology",
]
DOCTORS = {
    "General Medicine": "Dr. Rao", "Cardiology": "Dr. Iyer", "Endocrinology": "Dr. Khan",
    "Orthopedics": "Dr. Das", "Pediatrics": "Dr. Patel", "Dermatology": "Dr. Reddy",
}
