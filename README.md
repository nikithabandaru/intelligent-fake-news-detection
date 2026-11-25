# Intelligent Fake News Detection Using NLP and Machine Learning

An end-to-end full-stack web application and machine learning system that analyzes news articles using Natural Language Processing (NLP) and Supervised Machine Learning to classify them as **Real** or **Fake**.

---

## 📌 Project Overview

With the rapid dissemination of digital information, distinguishing between legitimate journalism and deceptive misinformation is a critical challenge. This project implements a production-ready Fake News Detection system leveraging the **ISOT Fake News Dataset** (University of Victoria).

The application extracts textual features via **TF-IDF Vectorization**, evaluates multiple classification algorithms (e.g., Logistic Regression, Passive Aggressive Classifier, Naive Bayes), and exposes an inference REST API built with **Flask** and **SQLite**. The user interface is built using **React** with dynamic visualization powered by **Chart.js**.

---

## 🛠️ Technology Stack

- **Machine Learning & NLP**: Python, Scikit-learn, NLTK, TF-IDF Vectorizer
- **Backend API**: Flask, Flask-CORS, SQLite
- **Frontend Application**: React, HTML5, CSS3, Chart.js
- **Data Analysis & Visualization**: Pandas, NumPy, Matplotlib, Seaborn
- **Development & Research**: Jupyter Notebooks

---

## 📁 Project Architecture

```text
Intelligent-Fake-News-Detection/
│
├── dataset/             # Original and processed datasets
│   ├── Fake.csv         # ISOT Fake news articles (23,481 records)
│   └── True.csv         # ISOT True news articles (21,417 records)
│
├── notebooks/           # Jupyter notebooks for EDA and model experimentation
├── backend/             # Flask REST API and SQLite database handlers
├── frontend/            # React user interface and Chart.js dashboards
├── models/              # Serialized ML models and vectorizers (joblib/pickle)
├── reports/             # Model evaluation metrics, charts, and confusion matrices
│
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore configuration
└── README.md            # Project documentation
```

---

## 📊 Dataset Summary

The system is trained and evaluated on the **ISOT Fake News Dataset**:
- **Fake Articles (`Fake.csv`)**: 23,481 articles across topics (News, Politics, Left-news, Government News, US_News, Middle-east).
- **True Articles (`True.csv`)**: 21,417 articles verified from Reuters across topics (PoliticsNews, Worldnews).
- **Total Corpus**: 44,898 articles.
- **Attributes**: `title`, `text`, `subject`, `date`.

---

## 🚀 Setup & Installation

### 1. Clone & Navigate
```bash
git clone <repo-url>
cd Intelligent-Fake-News-Detection
```

### 2. Python Environment Setup
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```
