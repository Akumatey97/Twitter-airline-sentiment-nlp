# Twitter Airline Sentiment Analysis (NLP with DistilBERT)

End-to-end NLP project using the **Twitter US Airline Sentiment** dataset to classify tweets about U.S. airlines as **positive**, **neutral**, or **negative**, and to understand key drivers of customer dissatisfaction.

This project is designed to showcase practical **data analyst → junior data scientist** skills, combining:
- Exploratory data analysis (EDA)
- Traditional ML baseline (TF–IDF + Logistic Regression)
- Modern transformer-based NLP (DistilBERT)
- Business-focused insights and visuals

---

## 1. Project Overview

Airlines receive thousands of customer tweets every day. Automatically classifying sentiment and extracting common complaint themes can help:
- Monitor customer satisfaction in real time
- Prioritize operational improvements (delays, cancellations, customer service, etc.)
- Support marketing and CX teams with data-driven insights

**Goal:**  
Build and compare models that classify airline-related tweets into sentiment categories and use the results to generate actionable recommendations for airline customer experience teams.

---

## 2. Dataset

- **Name:** Twitter US Airline Sentiment  
- **Size:** ~14,640 tweets about major U.S. airlines  
- **Labels:** `positive`, `neutral`, `negative` sentiment, plus `negative_reason` for many negative tweets  
- **Source:**  Kaggle 

> Note: Raw tweets may be subject to platform terms of service. This repo includes code and, if needed, only processed/derived data, not raw tweet text.

---

## 3. Repository Structure

```text
twitter-airline-sentiment-nlp/
│
├── data/
│   ├── raw/                # Original dataset (not committed if license restrictive)
│   └── processed/          # Cleaned / subset CSVs used in notebooks
│
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis & cleaning
│   ├── 02_traditional_models.ipynb   # TF–IDF + Logistic Regression baseline
│   └── 03_distilbert_model.ipynb     # DistilBERT fine-tuning and evaluation
│
├── src/                    # (Optional) Reusable Python modules (preprocessing, training, utils)
│
├── reports/
│   ├── figures/            # Exported plots (class distribution, confusion matrices, etc.)
│
└── README.md
