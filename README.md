# AI Healthcare Management Platform

An AI-powered healthcare management platform built with **Python and Streamlit**, combining patient management, appointment scheduling, predictive analytics, machine learning, and an AI health assistant in a single application.

> **Note:** This project uses fully synthetic healthcare data and is developed for educational purposes. It is not a medical device or a clinical decision-making system.

## Features

### 📊 Healthcare Dashboard

* Patient and appointment KPIs
* Age and readmission analysis
* Department-wise no-show analysis
* Monthly appointment trends
* Correlation analysis

### 👤 Patient Management

* Search and view patient records
* Individual readmission risk prediction
* Register new patients
* SQLite database integration using SQLAlchemy

### 📅 Appointment Management

* Search and filter appointments
* Upcoming no-show watchlist
* Appointment booking
* Predicted no-show risk

### 🤖 Machine Learning

* 30-day readmission risk prediction using **XGBoost**
* Appointment no-show risk prediction
* Feature importance analysis
* Automated model training and inference

### 🩺 AI Health Assistant

* Retrieval-based healthcare information
* TF-IDF knowledge-base retrieval
* LangChain integration
* Google Gemini integration
* Offline fallback responses
* Emergency keyword detection

The assistant provides general health information and does not diagnose medical conditions or provide medical treatment.

## Tech Stack

| Category         | Technologies             |
| ---------------- | ------------------------ |
| Language         | Python 3.11              |
| Application      | Streamlit                |
| Database         | SQLite, SQLAlchemy       |
| Data Analysis    | Pandas, NumPy            |
| Machine Learning | Scikit-learn, XGBoost    |
| AI / NLP         | LangChain, Google Gemini |
| Visualization    | Matplotlib, Seaborn      |

## Project Structure

```text
AI-Healthcare-Management-Platform/
│
├── app.py
├── requirements.txt
├── requirements.lock.txt
├── .env.example
│
├── data/
│   └── healthcare.db
│
├── src/
│   ├── config.py
│   ├── db.py
│   ├── data_gen.py
│   ├── models.py
│   ├── assistant.py
│   ├── analytics.py
│   └── bootstrap.py
│
└── scripts/
    └── train.py
```

## Machine Learning

The platform contains two predictive ML workflows:

**Readmission Risk Prediction**

An XGBoost model predicts the probability of patient readmission within 30 days and provides feature-importance insights.

**Appointment No-Show Prediction**

A machine learning model predicts the likelihood of an appointment being missed, helping identify appointments that may require additional attention.

> The models are trained on synthetic data. Their metrics demonstrate the functionality of the machine learning pipeline and should not be considered clinical performance.

## AI Assistant

The AI Assistant uses a retrieval-augmented approach to provide healthcare-related information.

The workflow includes:

1. TF-IDF-based retrieval from a healthcare knowledge base
2. Relevant information retrieval using LangChain
3. Gemini-powered response generation
4. Offline fallback when the Gemini API is unavailable
5. Emergency keyword detection

## Getting Started

### Prerequisites

* Python 3.11.9
* VS Code
* Git

### Installation

Clone the repository:

```bash
git clone https://github.com/mandaneha/AI-Healthcare-Management-Platform.git
cd AI-Healthcare-Management-Platform
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file from the example:

```bash
copy .env.example .env
```

For macOS/Linux:

```bash
cp .env.example .env
```

Add your Gemini API key to enable AI-generated responses:

```text
GEMINI_API_KEY=your_api_key_here
```

### Run the Application

```bash
streamlit run app.py
```

On the first launch, the application automatically:

* Creates the SQLite database
* Generates 1,000 synthetic patient records
* Generates 3,000 synthetic appointment records
* Trains the machine learning models

### Retrain Models

```bash
python -m scripts.train
```

To generate a locked dependency file:

```bash
pip freeze > requirements.lock.txt
```

## Data

All patient and appointment information is **synthetically generated** by `src/data_gen.py`.

No real patient information is required or included in this project.

The synthetic nature of the dataset means that model performance should be interpreted as a demonstration of the ML workflow rather than real-world healthcare performance.

## Disclaimer

This project is developed for **educational and demonstration purposes only**.

The machine learning models are trained on synthetic data, and the AI Assistant provides general informational responses. This application should not be used for medical diagnosis, treatment decisions, or clinical decision-making.
