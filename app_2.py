import pickle
import streamlit as st
import numpy as np
# Loading the model

with open('first_student_performance_model.pkl', 'rb') as file:
    model= pickle.load(file)

# streamlit UI 

st.title('Student Performance Prediction App')
st.write('This app predicts the student performance')
st.write('Please input the following parameter:')

math_score = st.number_input('Math Score', min_value=0, max_value=100, step=1)
reading_score = st.number_input('Reading Score', min_value=0, max_value=100, step=1)
writing_score = st.number_input('Writing Score', min_value=0, max_value=100, step=1)
gender_male_female = st.number_input('Gender', min_value=0, max_value=1, step=1)

# Prediction

if st.button('Predict'):
    user_input = np.array([[math_score, reading_score, writing_score, gender_male_female]])
    prediction = model.predict(user_input)
    
    performance_mapping = {0: 'A - Excellent',1: 'B - Very Good',2: 'C - Good',3: 'D - Pass',4: 'F - Fail'}
    
    predicted_student_performance = performance_mapping.get(int(prediction[0]), 'unknown')
    st.write(f'The predicted performance is: {predicted_student_performance}')

    # Footer

st.write('Made with Streamlit')


