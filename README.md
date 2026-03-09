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

--

## 3. EDA Findings
Exploratory analysis reveals that the dataset is highly imbalanced, with negative tweets dominating overall and for every airline. Most complaints are about customer service, late or cancelled flights, and lost luggage, while positive tweets are relatively rare. Tweets are typically between 80–140 characters long, which informed preprocessing and model design choices for the later sentiment classification models.

## 4. Baseline models summary
Train/validation/test setup

Used the cleaned tweet text and airline_sentiment labels on an 80/10/10 train/validation/test split, stratified by sentiment to preserve class proportions.

Text was lowercased and lightly cleaned (URLs, mentions removed), then vectorized with a TF–IDF representation using unigrams and bigrams.

Logistic Regression (TF–IDF, class_weight='balanced')
To handle sentiment imbalance (many more negative tweets), a multinomial Logistic Regression model was trained with class_weight='balanced' so minority classes (neutral, positive) receive higher weight.

Validation: accuracy ≈ 0.74, macro F1 ≈ 0.70.

Test: accuracy ≈ 0.78, macro F1 ≈ 0.73.

Per‑class test scores:

Negative: precision 0.88, recall 0.82, F1 0.85 (model captures most negative tweets).

Neutral: precision 0.57, recall 0.68, F1 0.62 (much improved compared to the unweighted baseline).

Positive: precision 0.72, recall 0.73, F1 0.72 (balanced precision/recall on the minority positive class).

These results show that class weighting trades a bit of negative‑class precision for significantly better recall and F1 on neutral and positive tweets, which is desirable when minority classes are important.

Linear SVM (TF–IDF)
A Linear SVM classifier was also trained on the same TF–IDF features for comparison.

Test: accuracy ≈ 0.77, macro F1 ≈ 0.70.

Per‑class test scores:

Negative: precision 0.83, recall 0.89, F1 0.86.

Neutral: precision 0.61, recall 0.53, F1 0.57.

Positive: precision 0.73, recall 0.63, F1 0.68.

SVM slightly favors the negative class with higher recall, but underperforms class‑weighted Logistic Regression on neutral and positive tweets, so Logistic Regression remains the primary traditional baseline.

Takeaways
Simple TF–IDF + linear models already achieve ~77–78% test accuracy and macro F1 between 0.70 and 0.73 on this dataset.

Class weighting in Logistic Regression significantly improves performance on neutral and positive sentiment without sacrificing overall accuracy, making it a strong baseline.

These limitations motivate using a transformer model (DistilBERT) better to capture context in short, ambiguous airline tweets and further improve minority‑class performance.

## 5. DistilBERT fine‑tuned model
To capture richer context in tweets than TF–IDF can provide, I fine‑tuned a DistilBERT model (distilbert-base-uncased) for 3‑class sentiment classification (negative, neutral, positive).
​

Setup:

Initialized DistilBertForSequenceClassification with 3 labels and a randomly initialized classification head.

Used the same 80/10/10 train/validation/test split as the traditional models, with text tokenized to a max length of 64 tokens.

Trained for 3 epochs with AdamW (learning rate 2e‑5, batch size 16) using the Hugging Face Trainer API and monitored validation metrics each epoch.
​
​

Test performance:

Accuracy: 0.84

Macro F1: ≈ 0.79

Per‑class test scores:

Negative: precision 0.89, recall 0.93, F1 0.91 (very strong on the majority class).

Neutral: precision 0.73, recall 0.65, F1 0.69 (substantial improvement vs TF–IDF baselines).

Positive: precision 0.77, recall 0.76, F1 0.77 (best balance across all models).

Compared to the best TF–IDF + Logistic Regression baseline (test accuracy ≈ 0.78, macro F1 ≈ 0.73), DistilBERT improves both overall accuracy and macro F1, particularly by better handling neutral and positive tweets that require more nuanced context understanding.

## 6. Model Comparison
| Model                                | Accuracy | Macro F1 |
|--------------------------------------|----------|----------|
| TF–IDF + Logistic Regression (bal.)  | 0.78     | 0.73     |
| TF–IDF + Linear SVM                  | 0.77     | 0.70     |
| DistilBERT (fine‑tuned)              | 0.84     | 0.79     |


## 7. Repository Structure

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


