"""AI-based Healthcare Management Platform (AIML-D). Run:  streamlit run app.py
All data is synthetic. This is a student project, not a medical device."""
import pandas as pd
import streamlit as st

from src import analytics, assistant, db
from src.bootstrap import bootstrap
from src.config import DEPARTMENTS, DOCTORS, GEMINI_API_KEY, NOSHOW_MODEL_PATH, RISK_MODEL_PATH
from src.models import load_bundle, predict_noshow, predict_risk, risk_level

st.set_page_config(page_title="Healthcare Management Platform", page_icon="🏥", layout="wide")


@st.cache_resource(show_spinner="Setting up database and training models (first run only)...")
def setup():
    bootstrap()
    return True


setup()


def models():
    return load_bundle(RISK_MODEL_PATH), load_bundle(NOSHOW_MODEL_PATH)


risk_bundle, noshow_bundle = models()
patients, appts = db.read_patients(), db.read_appointments()

st.sidebar.title("🏥 Healthcare Platform")
page = st.sidebar.radio("Go to", ["Dashboard", "Patients", "Appointments", "Risk Prediction", "AI Assistant"])
st.sidebar.caption("Synthetic data only. Not for real clinical use.")


# ---------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.title("Hospital Dashboard")
    done = appts[appts["status"].isin(["Completed", "No-Show"])]
    c = st.columns(5)
    c[0].metric("Patients", f"{len(patients):,}")
    c[1].metric("Appointments", f"{len(appts):,}")
    c[2].metric("Upcoming", int((appts["status"] == "Scheduled").sum()))
    c[3].metric("No-show rate", f"{(done['status'] == 'No-Show').mean():.1%}")
    c[4].metric("30-day readmission", f"{patients['readmitted_30d'].mean():.1%}")
    a, b = st.columns(2)
    a.pyplot(analytics.age_distribution(patients))
    b.pyplot(analytics.noshow_by_department(appts))
    a.pyplot(analytics.monthly_volume(appts))
    b.pyplot(analytics.correlation_heatmap(patients))


# ----------------------------------------------------------------- Patients
elif page == "Patients":
    st.title("Patients")
    q = st.text_input("Search by name or ID")
    view = patients
    if q:
        view = patients[patients["name"].str.contains(q, case=False) | (patients["id"].astype(str) == q.strip())]
    if risk_bundle is not None and len(view):
        view = view.assign(risk=predict_risk(risk_bundle, view).round(3))
        view["risk_level"] = view["risk"].map(risk_level)
    st.caption(f"{len(view):,} patients")
    st.dataframe(view.drop(columns=["readmitted_30d"]), use_container_width=True, hide_index=True)

    with st.expander("➕ Register a new (synthetic) patient"):
        with st.form("new_patient", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Name", "Patient New")
            age = c2.number_input("Age", 0, 120, 40)
            gender = c3.selectbox("Gender", ["Female", "Male", "Other"])
            bmi = c1.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
            bp = c2.number_input("Systolic BP", 70, 250, 120)
            glu = c3.number_input("Glucose (mg/dL)", 40, 500, 95)
            chol = c1.number_input("Cholesterol (mg/dL)", 80, 400, 180)
            smoker = c2.checkbox("Smoker")
            nc = c3.number_input("Chronic conditions", 0, 10, 0)
            pa = c1.number_input("Prior admissions", 0, 20, 0)
            stay = c2.number_input("Avg stay (days)", 0.0, 30.0, 0.0, 0.5)
            if st.form_submit_button("Save patient"):
                new_id = db.add_patient(name=name, age=int(age), gender=gender, bmi=bmi, systolic_bp=int(bp),
                                        glucose=int(glu), cholesterol=int(chol), smoker=int(smoker),
                                        num_conditions=int(nc), prior_admissions=int(pa), avg_stay_days=stay)
                st.success(f"Saved patient #{new_id}")
                st.rerun()


# ------------------------------------------------------------- Appointments
elif page == "Appointments":
    st.title("Appointments")
    tab1, tab2, tab3 = st.tabs(["All appointments", "No-show watchlist", "Book appointment"])

    with tab1:
        f1, f2 = st.columns(2)
        status = f1.multiselect("Status", sorted(appts["status"].unique()), default=["Scheduled"])
        dept = f2.multiselect("Department", DEPARTMENTS)
        v = appts[appts["status"].isin(status)] if status else appts
        if dept:
            v = v[v["department"].isin(dept)]
        st.dataframe(v.sort_values("scheduled_for"), use_container_width=True, hide_index=True)

    with tab2:
        up = appts[appts["status"] == "Scheduled"].copy()
        if noshow_bundle is None or up.empty:
            st.info("No upcoming appointments or model not trained yet.")
        else:
            up["no_show_risk"] = predict_noshow(noshow_bundle, up, patients).round(3)
            st.caption("Highest-risk upcoming appointments: good candidates for a reminder call or SMS.")
            st.dataframe(up.sort_values("no_show_risk", ascending=False).head(25),
                         use_container_width=True, hide_index=True)

    with tab3:
        with st.form("book"):
            c1, c2, c3 = st.columns(3)
            pid = c1.number_input("Patient ID", 1, int(patients["id"].max()), 1)
            dept = c2.selectbox("Department", DEPARTMENTS)
            when = c3.date_input("Date", pd.Timestamp.today() + pd.Timedelta(days=7))
            hour = c1.slider("Hour", 9, 17, 10)
            reminder = c2.checkbox("Send reminder", True)
            prior_ns = c3.number_input("Prior no-shows", 0, 10, 0)
            if st.form_submit_button("Book"):
                if int(pid) not in set(patients["id"]):
                    st.error("Unknown patient ID.")
                else:
                    sched = pd.Timestamp(when) + pd.Timedelta(hours=hour)
                    lead = max((sched.normalize() - pd.Timestamp.today().normalize()).days, 0)
                    row = dict(patient_id=int(pid), department=dept, doctor=DOCTORS[dept],
                               scheduled_for=sched.to_pydatetime(), lead_days=lead,
                               reminder_sent=int(reminder), prior_no_shows=int(prior_ns), status="Scheduled")
                    aid = db.add_appointment(**row)
                    msg = f"Booked appointment #{aid} with {DOCTORS[dept]}."
                    if noshow_bundle is not None:
                        p = predict_noshow(noshow_bundle, pd.DataFrame([row]), patients)[0]
                        msg += f" Predicted no-show risk: {p:.0%}."
                    st.success(msg)


# ---------------------------------------------------------- Risk Prediction
elif page == "Risk Prediction":
    st.title("30-day Readmission Risk")
    if risk_bundle is None:
        st.error("Risk model not found. Run `python -m scripts.train`.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 1, 95, 55)
    bmi = c2.slider("BMI", 14.0, 45.0, 27.0, 0.1)
    bp = c3.slider("Systolic BP", 90, 200, 130)
    glu = c1.slider("Glucose (mg/dL)", 65, 350, 110)
    chol = c2.slider("Cholesterol (mg/dL)", 100, 350, 200)
    smoker = c3.checkbox("Smoker")
    nc = c1.slider("Chronic conditions", 0, 8, 2)
    pa = c2.slider("Prior admissions", 0, 10, 1)
    stay = c3.slider("Average stay (days)", 0.0, 20.0, 4.0, 0.5)
    row = pd.DataFrame([dict(age=age, bmi=bmi, systolic_bp=bp, glucose=glu, cholesterol=chol, smoker=int(smoker),
                             num_conditions=nc, prior_admissions=pa, avg_stay_days=stay)])
    p = float(predict_risk(risk_bundle, row)[0])
    lvl = risk_level(p)
    st.subheader(f"Risk: {p:.1%}  ({lvl})")
    st.progress(min(p, 1.0))
    (st.success if lvl == "Low" else st.warning if lvl == "Medium" else st.error)(
        {"Low": "Routine follow-up is likely enough.",
         "Medium": "Consider a follow-up call within a week of discharge.",
         "High": "Flag for care-team review and a structured discharge plan."}[lvl])
    st.pyplot(analytics.importance_chart(risk_bundle["importances"]))
    st.caption(f"Model: XGBoost, test ROC-AUC {risk_bundle['metrics']['roc_auc']} on synthetic data. "
               "Decision-support demo only.")


# -------------------------------------------------------------- AI Assistant
else:
    st.title("AI Health Assistant")
    st.caption("Gemini mode is on." if GEMINI_API_KEY else
               "Offline mode: add GEMINI_API_KEY to .env for full AI answers. General information only.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, text in st.session_state.chat:
        st.chat_message(role).markdown(text)
    if prompt := st.chat_input("Ask a general health question..."):
        st.chat_message("user").markdown(prompt)
        with st.spinner("Thinking..."):
            reply = assistant.answer(prompt, st.session_state.chat)
        st.chat_message("assistant").markdown(reply)
        st.session_state.chat += [("user", prompt), ("assistant", reply)]
