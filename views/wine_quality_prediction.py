import streamlit as st
from util import *
from models import *


# Function to display the model prediction and probability

def display_prediction(pred, prob):
    st.subheader("Prediction")
    if pred[0] == 0:
        prob = "{:.3f}".format(prob[0][0])
        st.success(f"Heart Disease: NO")
    else:
        prob = "{:.3f}".format(prob[0][0])
        st.error(f"Heart Disease: YES")


# Application title and description
st.title('WineWise🍷')
st.write(":blue[***Discover the quality of your wine with a click 🥂🔍***]")
st.write("WineWise analyzes your input data to predict the quality of wine. Using advanced algorithms, it provides "
         "quick and reliable assessments, helping you choose the perfect bottle for any occasion 🍷✨")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset).', icon='ℹ️')

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = wine_quality_input_parameters()
