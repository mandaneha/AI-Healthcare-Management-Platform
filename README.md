# AIML-D: AI-based Healthcare Management Platform

Streamlit app with patient records, appointment scheduling, two ML models and an AI health assistant.
**All data is synthetic** (generated in `src/data_gen.py`). Student project, not a medical device.

## Run in VS Code (Python 3.11.9)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # optional: add GEMINI_API_KEY for full AI answers
streamlit run app.py
```
First launch creates `data/healthcare.db`, seeds 1,000 patients and 3,000 appointments, and trains both models.
Retrain any time with `python -m scripts.train`. Lock versions with `pip freeze > requirements.lock.txt`.

## Features
| Page | What it does |
|---|---|
| Dashboard | KPIs, age/readmission chart, no-show by department, monthly volume, correlation heatmap |
| Patients | Search, per-patient readmission risk, register new patient (SQLAlchemy + SQLite) |
| Appointments | Filter, no-show watchlist for upcoming visits, booking with predicted no-show risk |
| Risk Prediction | XGBoost 30-day readmission risk with feature importance |
| AI Assistant | LangChain TF-IDF retrieval over a small knowledge base + Gemini (google-genai); offline fallback; emergency keyword guard |

## Structure
```
app.py              Streamlit UI
src/config.py       paths, env vars        src/db.py         SQLAlchemy models + queries
src/data_gen.py     synthetic data         src/models.py     XGBoost + sklearn training/inference
src/assistant.py    RAG + Gemini           src/analytics.py  matplotlib/seaborn charts
src/bootstrap.py    first-run setup        scripts/train.py  retrain models
```

## Notes
- Models learn from synthetic labels, so metrics only show the pipeline works.
- The assistant gives general information only and never diagnoses.
- To go further: replace the knowledge base with vetted documents, add login/roles, and swap SQLite for PostgreSQL via `DATABASE_URL`.
