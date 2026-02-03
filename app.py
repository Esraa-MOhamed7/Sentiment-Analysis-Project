import streamlit as st
import joblib

model_loaded = joblib.load("sentiment_model.pkl")
vectorizer_loaded = joblib.load("vectorizer.pkl")

def predict_sentiment(text: str):
    text_vec = vectorizer_loaded.transform([text])
    pred = model_loaded.predict(text_vec)[0]
    return "Positive" if pred == 1 else "Negative"

st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #39B34A;'>Sentiment Analysis Web App</h1>",
    unsafe_allow_html=True
)

st.image("https://i.pinimg.com/736x/1c/7d/2f/1c7d2fbeea7ede56294e5a3b933f9638.jpg")

st.markdown(
    "<p style='text-align: center; color: #555;'>Analyze reviews instantly with AI-powered sentiment detection</p>",
    unsafe_allow_html=True
)

user_input = st.text_area("Your Review:", placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.markdown(
            f"<p style='text-align: center; color: #39B34A; font-size:22px; font-weight:bold;'>Sentiment: {sentiment}</p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter some text before analyzing.")

# Prediction history
if "history" not in st.session_state:
    st.session_state["history"] = []

if user_input.strip():
    st.session_state["history"].append((user_input, predict_sentiment(user_input)))

if st.session_state["history"]:
    st.subheader("📜 Prediction History")
    for text, sent in st.session_state["history"]:
        st.write(f"**{text}** → {sent}")
