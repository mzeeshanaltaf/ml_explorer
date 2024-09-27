import streamlit as st
import pandas as pd
from util import *
from models import *


# Function to display the model prediction and probability

def display_prediction(pred, prob):
    st.subheader("Prediction")
    df_prob = pd.DataFrame(prob)
    df_prob.columns = ['No Parkinson Disease', 'Parkinson Disease']
    if pred[0] == 0:
        prob = "{:.3f}".format(prob[0][0])
        st.success(f"Parkinson Disease: NO")
    else:
        prob = "{:.3f}".format(prob[0][0])
        st.error(f"Parkinson Disease: YES")

    st.subheader('Prediction Probability')
    st.dataframe(df_prob,
                 column_config={
                     'No Parkinson Disease': st.column_config.ProgressColumn(
                         'No Parkinson Disease',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                     'Parkinson Disease': st.column_config.ProgressColumn(
                         'Parkinson Disease',
                         format='%.2f',
                         width='medium',
                         min_value=0,
                         max_value=1
                     ),
                 }, hide_index=True)


# Application title and description
st.title('NeuroTrack️🧠')
st.write(":blue[***Tracking early signs of Parkinson’s 🧠🔍***]")
st.write(
    "NeuroTrack helps users assess the risk of Parkinson’s disease by analyzing their input data. Powered by "
    "advanced machine learning, it offers quick and accurate predictions to keep you informed and proactive about "
    "your neurological health 🌟🩺")
st.info('Dataset for this app is taken from '
        '[Kaggle](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set).', icon='ℹ️')

# Show Disclaimer
with st.expander('Disclaimer', icon=':material/info:'):
    display_disclaimer()

# Input Parameters
st.subheader('Input Parameters')
with st.expander('User Input', expanded=True, icon=':material/settings_input_component:'):
    input_data = parkinson_input_parameters()

# Machine Learning Model Selection
st.subheader('Choose Machine Learning Model')
model_name = st.selectbox('Select the Model', supported_models, label_visibility="collapsed")

# Train the model and get prediction and probability of outcome
model, scalar, df_performance_metric, cm = train_model(model_name, 'Parkinson', 'binary')
prediction, probability = model_predictions(input_data, model, scalar)
display_prediction(prediction, probability)

# Display performance metrics
display_performance_metrics(df_performance_metric)

# Display Confusion Matrix
labels = ['Disease', 'Not Disease']
display_confusion_matrix(cm, labels)
