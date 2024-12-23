import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('trained_model.sav', 'rb'))
st.title('Iris Prediction')

sepel_len= st.number_input('sepel lenght')
spel_w= st.number_input('sepel width')
patel_len= st.number_input('patel lenght')
patel_w= st.number_input('patel width')

input_data =[sepel_len,spel_w,patel_len,patel_w]

result = ''
if st.button('Result'):
    input_data = np.asarray(input_data).reshape(1,-1)
    result = model.predict(input_data)[0]
st.success(result)
