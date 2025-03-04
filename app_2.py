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
final_score = st.number_input('Final Score', min_value=0, max_value=100, step=1)

# Prediction

if st.button('Predict'):
    user_input = np.array([[math_score, reading_score, writing_score, final_score]])
    prediction = model.predict(user_input)
    species_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    
    predicted_student_performance = species_mapping.get(int(prediction[0]), 'unknown')
    st.write(f'The predicted species is: {predicted_student_performance}')

    # Footer

st.write('Made with Streamlit')


