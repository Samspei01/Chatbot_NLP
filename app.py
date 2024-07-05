import streamlit as st
import keras.models
import pickle
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from colorama import Fore, Style

# Load the necessary files
with open("json.json") as file:  
    data = json.load(file)

model = keras.models.load_model('chat_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 31

def get_response(inp):
    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]),
                                         truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    return "I'm not sure how to respond to that."

# Streamlit UI
st.title("Chatbot")

input_text = st.text_input("You: ", "")
if st.button("Send"):
    response = get_response(input_text)
    st.text_area("ChatBot:", value=response, height=100, max_chars=None, key=None)
    
# run with : streamlit run app.py