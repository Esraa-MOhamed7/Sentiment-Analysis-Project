# 💬 Sentiment Analysis — ML Model + Automated Workflow

> What if your model didn't just predict — but actually responded?

This project connects a custom-trained **Sentiment Analysis model** with an **automated n8n workflow** that handles the business logic: thanking happy customers, apologizing to unhappy ones, and alerting the manager — all without manual intervention.

---

## How It Works

### The Workflow (n8n)

1. User submits a review via a **form**
2. The model classifies it as **Positive** or **Negative**
3. Result + data get logged in **Google Sheets** with timestamp
4. **If Positive** → automatic thank-you message sent to the user
5. **If Negative** → automatic apology sent + **Alert fired to the Manager**

Two scenarios are demonstrated in the video: a positive case and a negative case.

---

## The Model

Built and trained from scratch using **PyTorch**:

| Component | Detail |
|---|---|
| Baseline | Logistic Regression |
| Deep Learning Model | LSTM (PyTorch) |
| Output | Sentiment label + confidence score |
| Saved With | `joblib` (model + vectorizer) |

### Training Pipeline

- Trained and evaluated on a labeled reviews dataset
- Compared LSTM accuracy against Logistic Regression baseline
- Saved model and vectorizer for reuse without retraining
- Built a prediction function returning both label and confidence score

---

##  Web App (Streamlit)

A lightweight interface to test the model directly:

- Text input for any review
- Analyze button → returns Positive / Negative
- Confidence progress bar
- Emoji-enhanced results
- Prediction history within the session

### Run Locally

```bash
git clone https://github.com/Esraa-MOhamed7/Sentiment-Analysis-Project.git
cd Sentiment-Analysis-Project
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model Training | PyTorch (LSTM) |
| Baseline | Scikit-learn (Logistic Regression) |
| Serialization | joblib |
| Web App | Streamlit |
| Workflow Automation | n8n |
| Data Logging | Google Sheets |
| Alerts | Manager notification via n8n |

---

## 🔗 Links

- [Code & Workflow](https://github.com/Esraa-MOhamed7/Sentiment-Analysis-Project)
- [Kaggle Notebook](https://www.kaggle.com/code/esraamoh7med/from-reviews-to-predictions-sentiment-analysis)

---

## 🙏Note

This project was an experiment in connecting Machine Learning with Automation and real Business Logic — where the model's output doesn't just sit in a notebook, but actually triggers meaningful actions.

Feedback and suggestions are always welcome!
