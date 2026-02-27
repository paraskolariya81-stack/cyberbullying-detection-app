import streamlit as st
from model import predict

st.title("Cyberbullying Detection App")

user_input = st.text_area("Enter Text")

if st.button("Predict"):
    result = predict(user_input)
    st.success(f"Prediction: {result}")
