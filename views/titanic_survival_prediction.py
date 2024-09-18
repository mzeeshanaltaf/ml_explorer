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
st.title('TitanicSurvive')
st.write(":blue[***Uncover the fate of Titanic passengers 🚢🔍***]")
st.write("TitanicSurvive uses historical passenger data to predict the likelihood of survival from the Titanic "
         "disaster. With a simple input, it provides insights into whether a passenger survived, using advanced "
         "algorithms to analyze and interpret the data 🛳️📊")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/competitions/titanic/data).', icon='ℹ️')

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = titanic_input_parameters()
