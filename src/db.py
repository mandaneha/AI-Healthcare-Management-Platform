"""SQLAlchemy 2.0 persistence layer (SQLite by default)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .config import DB_URL
from .data_gen import generate_appointments, generate_patients


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(10))
    bmi: Mapped[float] = mapped_column(Float)
    systolic_bp: Mapped[int] = mapped_column(Integer)
    glucose: Mapped[int] = mapped_column(Integer)
    cholesterol: Mapped[int] = mapped_column(Integer)
    smoker: Mapped[int] = mapped_column(Integer, default=0)
    num_conditions: Mapped[int] = mapped_column(Integer, default=0)
    prior_admissions: Mapped[int] = mapped_column(Integer, default=0)
    avg_stay_days: Mapped[float] = mapped_column(Float, default=0.0)
    readmitted_30d: Mapped[int] = mapped_column(Integer, default=0)


class Appointment(Base):
    __tablename__ = "appointments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    department: Mapped[str] = mapped_column(String(40))
    doctor: Mapped[str] = mapped_column(String(40))
    scheduled_for: Mapped[datetime] = mapped_column(DateTime)
    lead_days: Mapped[int] = mapped_column(Integer, default=0)
    reminder_sent: Mapped[int] = mapped_column(Integer, default=0)
    prior_no_shows: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(15), default="Scheduled")


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, future=True)
    return _engine


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def seed_if_empty(n_patients: int = 1000, n_appointments: int = 3000) -> bool:
    """Fill empty tables with synthetic data. Returns True if seeding happened."""
    eng = get_engine()
    with Session(eng) as s:
        if s.scalar(select(func.count(Patient.id))):
            return False
    patients = generate_patients(n_patients)
    patients.to_sql("patients", eng, if_exists="append", index=False)
    generate_appointments(patients, n_appointments).to_sql("appointments", eng, if_exists="append", index=False)
    return True


def read_patients() -> pd.DataFrame:
    return pd.read_sql(select(Patient), get_engine())


def read_appointments() -> pd.DataFrame:
    return pd.read_sql(select(Appointment), get_engine(), parse_dates=["scheduled_for"])


def add_patient(**fields) -> int:
    with Session(get_engine()) as s:
        p = Patient(**fields)
        s.add(p)
        s.commit()
        return p.id


def add_appointment(**fields) -> int:
    with Session(get_engine()) as s:
        a = Appointment(**fields)
        s.add(a)
        s.commit()
        return a.id


def update_appointment_status(appointment_id: int, status: str) -> None:
    with Session(get_engine()) as s:
        a = s.get(Appointment, appointment_id)
        if a:
            a.status = status
            s.commit()
