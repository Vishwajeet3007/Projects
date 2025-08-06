import streamlit as st
import torch
from utils.load_model import load_model_and_tokenizer
from inference.beam_search import beam_search_predict

st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.title("🧠 Next Word Predictor")
st.write("Enter a sentence, and the model will predict the next few words using beam search.")

input_text = st.text_input("Enter text:", "Once upon a time")
beam_width = st.slider("Beam Width", min_value=1, max_value=10, value=3)
max_len = st.slider("Max Prediction Length", min_value=1, max_value=30, value=10)

if st.button("Predict"):
    with st.spinner("Generating predictions..."):
        model, tokenizer = load_model_and_tokenizer()
        results = beam_search_predict(model, tokenizer, input_text, beam_width=beam_width, max_len=max_len)

    st.subheader("🔮 Predictions:")
    for i, (text, score) in enumerate(results):
        st.markdown(f"**{i+1}.** {text} _(score: {score:.2f})_")
