import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('LinearRegression.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the fitted LabelEncoders
with open('label_encoder_name.pkl', 'rb') as le_file:
    le_name = pickle.load(le_file)

with open('label_encoder_genre.pkl', 'rb') as le_file:
    le_genre = pickle.load(le_file)

with open('label_encoder_director.pkl', 'rb') as le_file:
    le_director = pickle.load(le_file)

with open('label_encoder_actor 1.pkl', 'rb') as le_file:
    le_actor1 = pickle.load(le_file)

with open('label_encoder_actor 2.pkl', 'rb') as le_file:
    le_actor2 = pickle.load(le_file)

with open('label_encoder_actor 3.pkl', 'rb') as le_file:
    le_actor3 = pickle.load(le_file)

rating = pd.read_csv("Movie rating.csv")

# Streamlit app
st.title('Movie Rating Prediction App')
feature_names = ['Name', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

def encode_with_default(label_encoder, selected_value, default_value):
    if np.isin(selected_value, label_encoder.classes_):
        return label_encoder.transform([selected_value])[0]
    else:
        # Handle unseen label by returning a default value
        return default_value


# Input features
selected_name = st.selectbox('Movie Title:', rating['Name'].unique())
selected_genre = st.selectbox('Movie Genre:', rating['Genre'].unique())
selected_director = st.selectbox('Movie Director:', rating['Director'].unique())
selected_actor1 = st.selectbox('Movie Actor 1:', rating['Actor 1'].unique())
selected_actor2 = st.selectbox('Movie Actor 2:', rating['Actor 2'].unique())
selected_actor3 = st.selectbox('Movie Actor 3:', rating['Actor 3'].unique())

# Filter the dataset based on selected values
selected_movie = rating[
    (rating['Name'] == selected_name) &
    (rating['Genre'] == selected_genre) &
    (rating['Director'] == selected_director) &
    (rating['Actor 1'] == selected_actor1) &
    (rating['Actor 2'] == selected_actor2) &
    (rating['Actor 3'] == selected_actor3)
]
default_value = 0
# Encode the selected features using the loaded LabelEncoders with default value
encoded_name = encode_with_default(le_name, selected_name, default_value)
encoded_genre = encode_with_default(le_genre, selected_genre, default_value)
encoded_director = encode_with_default(le_director, selected_director, default_value)
encoded_actor1 = encode_with_default(le_actor1, selected_actor1, default_value)
encoded_actor2 = encode_with_default(le_actor2, selected_actor2, default_value)
encoded_actor3 = encode_with_default(le_actor3, selected_actor3, default_value)

# Create a DataFrame with the selected features
input_data = pd.DataFrame({
    'Name': [encoded_name],
    'Genre': [encoded_genre],
    'Director': [encoded_director],
    'Actor 1': [encoded_actor1],
    'Actor 2': [encoded_actor2],
    'Actor 3': [encoded_actor3]
})

# Ensure the columns are in the same order as during training
input_data = input_data[feature_names]

# Add the button here
if st.button("Predict Rating"):
    if not selected_movie.empty:
        actual_rating = selected_movie['Rating'].values[0]
        st.success(f'Movie Rating: {actual_rating}')
    else:
        st.warning('SELECTED MOVIE RATING NOT FOUND, CHECK YOUR INPUT.')