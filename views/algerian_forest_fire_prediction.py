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
st.title('FireGuard')
st.write(":blue[***Stay alert to prevent forest fires 🌲🔥***]")
st.write("FireGuard analyzes environmental data to predict the likelihood of forest fires. Using advanced predictive "
         "algorithms, it offers timely alerts and insights to help you stay prepared and protect natural areas from "
         "potential wildfires 🚒🌳")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset).', icon='ℹ️')

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = algerian_forest_fire_input_parameters()