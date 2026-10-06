"""Matplotlib / seaborn figures used by the Streamlit dashboard."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def age_distribution(patients: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(data=patients, x="age", hue="readmitted_30d", multiple="stack", bins=20, ax=ax,
                 palette={0: "#4C9BE8", 1: "#E8684C"})
    ax.set_title("Age distribution (orange = readmitted within 30 days)")
    fig.tight_layout()
    return fig


def noshow_by_department(appts: pd.DataFrame):
    done = appts[appts["status"].isin(["Completed", "No-Show"])]
    rate = (done.assign(ns=done["status"].eq("No-Show")).groupby("department")["ns"].mean()
            .sort_values() * 100)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    rate.plot.barh(ax=ax, color="#4C9BE8")
    ax.set_xlabel("No-show rate (%)")
    ax.set_title("No-show rate by department")
    fig.tight_layout()
    return fig


def monthly_volume(appts: pd.DataFrame):
    m = appts.set_index("scheduled_for").groupby([pd.Grouper(freq="MS"), "status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    m.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xticklabels([d.strftime("%b %y") for d in m.index], rotation=45)
    ax.set_xlabel("")
    ax.set_title("Appointments per month")
    fig.tight_layout()
    return fig


def correlation_heatmap(patients: pd.DataFrame):
    cols = ["age", "bmi", "systolic_bp", "glucose", "cholesterol", "num_conditions",
            "prior_admissions", "avg_stay_days", "readmitted_30d"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(patients[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=False)
    ax.set_title("Feature correlations")
    fig.tight_layout()
    return fig


def importance_chart(importances: dict):
    s = pd.Series(importances).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    s.plot.barh(ax=ax, color="#4C9BE8")
    ax.set_title("What drives readmission risk (model importance)")
    fig.tight_layout()
    return fig
