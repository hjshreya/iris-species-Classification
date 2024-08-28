import streamlit as st
import pickle
import numpy as np

st.set_page_config(layout="wide")

# Load the model
model_path = "/Users/shrutishreya/Desktop/iris_files/mlM.pkl"
model = pickle.load(open(model_path, "rb"))

# Streamlit app
st.title("Iris Flower Species Prediction")

st.image('/Users/shrutishreya/Desktop/flo.jpg', caption='', width=700)

st.markdown("<div style='margin-top: 50px;'></div>", 
unsafe_allow_html=True)

st.write("Enter the features below to predict the flower species:")

# Input fields for features
sepal_length = st.number_input("Sepal Length", format="%.2f")
sepal_width = st.number_input("Sepal Width", format="%.2f")
petal_length = st.number_input("Petal Length", format="%.2f")
petal_width = st.number_input("Petal Width", format="%.2f")

# Predict button

btn = st.button("Predict")

if btn:
    # Create feature array for prediction
    float_features = [sepal_length, sepal_width, petal_length, 
    petal_width]
    features = [np.array(float_features)]

    # Make prediction
    prediction = model.predict(features)

    species_mapping = {
        0: ("Setosa", "/Users/shrutishreya/Desktop/setosa.jpg"),
        1: ("Versicolor", "/Users/shrutishreya/Desktop/versicolour.jpeg"),
        2: ("Virginica", "/Users/shrutishreya/Desktop/virginica.jpeg")
    }
    
    species, image_path = species_mapping[prediction[0]] 

    # Show result
    st.write(f"The flower species is: {species}")
    st.image(image_path, caption=f"{species}")

st.markdown("<div style='margin-top: 50px;'></div>", 
unsafe_allow_html=True)
st.markdown("<div style='margin-top: 50px;'></div>", 
unsafe_allow_html=True)
st.markdown("<div style='margin-top: 50px;'></div>", 
unsafe_allow_html=True)
st.markdown("<div style='margin-top: 50px;'></div>", 
unsafe_allow_html=True)
st.markdown("<div style='margin-top: 50px;'></div>", 
unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h6 style='text-align: center;'>Created by Shreya Jha</h6>", 
unsafe_allow_html=True)

