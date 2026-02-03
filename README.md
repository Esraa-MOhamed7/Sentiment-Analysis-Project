# 📝 Sentiment Analysis Project

## Overview
This project demonstrates the complete workflow of building and deploying a **Sentiment Analysis system**.  
We started from raw data in a notebook, trained both baseline ML and deep learning models (using PyTorch), and then deployed the final model in a **Streamlit web application** with a modern, attractive UI.  
The app allows users to input product reviews and instantly see whether the sentiment is **Positive** or **Negative**, along with a confidence score and prediction history.

---

## Project Steps 

1. **Setup Project**
   - Imported necessary libraries.
   - Loaded the Amazon Reviews dataset (`train.ft.txt.bz2` and `test.ft.txt.bz2`).
   - Each line contained a label (`__label__1` = negative, `__label__2` = positive) followed by review text.

2. **Data Loading**
   - Loaded data into pandas DataFrames.
   - Converted labels into numerical format (0 = negative, 1 = positive).
   - Inspected samples to understand structure.

3. **Text Preprocessing**
   - Lowercased all text.
   - Removed punctuation and special characters.
   - Tokenized words.
   - Optionally removed stopwords.
   - Ensured consistency and reduced noise.

4. **Feature Extraction**
   - Applied **TF-IDF Vectorization** to convert text into numerical features.
   - Captured word importance across the dataset.

5. **Baseline ML Model**
   - Trained a **Logistic Regression classifier**.
   - Established baseline accuracy for comparison with deep learning models.

6. **Deep Learning with PyTorch**
   - Converted reviews into fixed-length sequences of word indices.
   - Mapped unknown words to `<unk>` and applied `<pad>` for padding.
   - Converted labels into tensors (0 = negative, 1 = positive).
   - Implemented a custom `Dataset` class and `DataLoader` for batching.

7. **LSTM Model**
   - Defined an LSTM-based sentiment classifier:
     - Embedding layer → converts word indices into dense vectors.
     - LSTM → captures sequence context.
     - Fully connected layer → outputs probability of positive sentiment.

8. **Training Loop with Validation, Early Stopping, and Scheduler**
   - Tracked training and validation accuracy each epoch.
   - Used **early stopping** to prevent overfitting.
   - Applied a **learning rate scheduler** to reduce LR when validation accuracy plateaued.

9. **Evaluation**
   - Evaluated trained model on the test set.
   - Computed accuracy and compared with baseline Logistic Regression.

10. **Saving the Trained Model**
    - Saved trained PyTorch model and vectorizer with `joblib`.
    - Ensured reusability without retraining.

11. **Loading the Saved Model**
    - Tested loading the model and vectorizer.
    - Verified predictions on new text samples.

12. **Prediction Function**
    - Built a function to predict sentiment for new text inputs.
    - Returned both sentiment label and confidence score.

13. **Web App Deployment (Streamlit)**
    - Created `app.py` with:
      - Input text box for reviews.
      - Button to analyze sentiment.
      - Display of results (Positive/Negative).
      - Confidence score progress bar.
      - Prediction history with emojis.
      - Gradient banner, sidebar info, and footer credits.

---

## Features
- End-to-end workflow: **Notebook → PyTorch Model → Web App**  
- Baseline ML model (Logistic Regression) + Deep Learning (LSTM with PyTorch)  
- Confidence progress bar for prediction strength  
- Emoji-enhanced results and history   

---

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-project.git
   cd sentiment-analysis-project

2. pip install -r requirements.txt
3. streamlit run app.py
