"""Health-information assistant: local retrieval (LangChain TF-IDF) + Gemini (google-genai).

Works offline without an API key by returning the retrieved reference notes.
This is general wellness information only, never a diagnosis.
"""
from __future__ import annotations

from functools import lru_cache

from .config import GEMINI_API_KEY, GEMINI_MODEL

KNOWLEDGE_BASE = [
    "High blood pressure (hypertension): often has no symptoms. Regular checks, less salt, regular exercise, a healthy weight, limiting alcohol and not smoking all help. Medicines must only be changed by a doctor.",
    "Type 2 diabetes: managed with balanced meals, physical activity, weight management, regular glucose monitoring and prescribed treatment. Warning signs include excess thirst, frequent urination and unexplained tiredness.",
    "Cholesterol: high LDL raises heart disease risk. Fibre, fruit, vegetables, nuts, fish and less fried or processed food help. A doctor can advise on testing frequency.",
    "BMI: a screening number (weight in kg divided by height in metres squared). About 18.5 to 24.9 is the usual healthy range for adults, but it does not capture muscle mass or fat distribution.",
    "Smoking: quitting lowers the risk of heart disease, stroke, lung disease and cancer within months to years. Counselling and nicotine replacement can double the chance of quitting successfully.",
    "Preparing for an appointment: list your symptoms and questions, bring current medicines and past reports, note when symptoms started, and arrive early. Confirm or cancel ahead of time so the slot can be reused.",
    "Hospital readmission: following discharge instructions, taking medicines as prescribed, attending follow-up visits and watching for warning signs lowers the chance of returning within 30 days.",
    "Healthy sleep and activity: adults generally do well with 7 to 9 hours of sleep and about 150 minutes of moderate exercise per week, unless a doctor advises otherwise.",
    "Fever in adults: rest, fluids and monitoring are usual first steps. See a doctor if it lasts more than 3 days, is very high, or comes with a stiff neck, confusion, rash or breathing trouble.",
]

SYSTEM_PROMPT = (
    "You are a careful healthcare information assistant inside a hospital management demo. "
    "Give clear, short, general wellness information in plain language. Never diagnose, never "
    "prescribe or give medicine doses, and always suggest seeing a qualified clinician for "
    "personal medical decisions. Use the reference notes when relevant. If symptoms sound "
    "serious, tell the user to seek emergency care."
)

RED_FLAGS = ["chest pain", "can't breathe", "cannot breathe", "difficulty breathing", "stroke",
             "unconscious", "severe bleeding", "suicid", "overdose", "seizure", "heart attack"]

EMERGENCY_MSG = (
    "**This may be an emergency.** Please contact your local emergency number or go to the nearest "
    "hospital right now. If you are thinking about harming yourself, reach out to local crisis "
    "services or someone you trust immediately. I can't assess emergencies in chat."
)


def is_emergency(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in RED_FLAGS)


@lru_cache(maxsize=1)
def _retriever():
    from langchain_community.retrievers import TFIDFRetriever
    return TFIDFRetriever.from_texts(KNOWLEDGE_BASE, k=3)


def retrieve(query: str, k: int = 3) -> list[str]:
    try:
        return [d.page_content for d in _retriever().invoke(query)][:k]
    except Exception:  # LangChain missing or failed: fall back to plain scikit-learn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english").fit(KNOWLEDGE_BASE + [query])
        sims = cosine_similarity(vec.transform([query]), vec.transform(KNOWLEDGE_BASE))[0]
        return [KNOWLEDGE_BASE[i] for i in sims.argsort()[::-1][:k] if sims[i] > 0]


def _offline(notes: list[str]) -> str:
    if not notes:
        return "I don't have a reference note for that. Please ask a clinician."
    body = "\n\n".join(f"- {n}" for n in notes)
    return f"Offline mode (no Gemini key set). Related reference notes:\n\n{body}\n\n_General information only, not medical advice._"


def answer(question: str, history: list[tuple[str, str]] | None = None) -> str:
    if is_emergency(question):
        return EMERGENCY_MSG
    notes = retrieve(question)
    if not GEMINI_API_KEY:
        return _offline(notes)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        contents = [types.Content(role="user" if r == "user" else "model", parts=[types.Part(text=t)])
                    for r, t in (history or [])[-6:]]
        ctx = "\n".join(f"- {n}" for n in notes)
        contents.append(types.Content(role="user", parts=[types.Part(
            text=f"Reference notes:\n{ctx}\n\nQuestion: {question}")]))
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.3,
                                               max_output_tokens=600))
        return (resp.text or _offline(notes)) + "\n\n_General information only, not medical advice._"
    except Exception as exc:
        return _offline(notes) + f"\n\n(Gemini unavailable: {type(exc).__name__})"
